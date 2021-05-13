//FIXME: test gradient implementation

//FIXME: implement gradients and rank constraints on dK d2K etc forms
//FIXME: to do this, need to fix yyycK2 implementation to call further down the tree for d..K..del.. gradients

//FIXME: kernel chains, pass pxyprod in
//FIXME: kernel inheritance for K1,K3,Km odd
//FIXME: complex kernels for K1,K3,Km odd
//FIXME: kernel8xx for K1,K3,Km odd
//FIXME: do magterm for m != 2 + everything other than the most basic LL2,dLL2 functionality




//
// Basic kernel class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//NB - to add new kernel definitions search for ADDHERE
//KERNELSHERE - labels where kernel is actually evaluated


#ifndef _mercer_h
#define _mercer_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "gentype.h"
#include "vector.h"
#include "sparsevector.h"
#include "matrix.h"
#include "numbase.h"
#include "awarestream.h"


#define DEFAULT_VECT_INDEX -4
#define VECINFOSCRATCHSIZE 4*DEFAULT_NUM_TUPLES


#define DEFAULT_NUMKERNSAMP 10



// Note on kernel numbering
//
// In the 0-999 range:
//
// 0  -99  are regular kernels (which may or may not be Mercer)
// 100-199 are 0/1 neural network kernels (monotonic increasing functions of
//         x'y typically but not always with outputs ranging 0 to 1)
// 200-299 are -1/+1 neural network kernels (monotonic increasing functions of
//         x/y typically but not always with outputs ranging -1 to 1)
// 300-399 distance kernels (return -1/2 ||x-y|| for different norms).  Used
//         by KNN for example.
//
// 400-449 monotonic density kernels of the form prod_k f(x_k-y_k), where f
//         is monotonic increasing, 0 < f(x) < 1, and f(0) = 1/2
// 450-499 monotonic density kernels of the form prod_k f(x_k-y_k), where f
//         is monotonic decreasing, 0 < f(x) < 1, and f(0) = 1/2
//         (as per 400-449 but with x/y order reversed)
//
// 500-549 monotonic dense derivatives of kernels 400-449 - that is,
//         prod_k f'(x_k-y_k), where f' is the derivative of monotonic function
//         f for corresponding monotonic density kernel
// 550-599 monotonic dense derivatives of kernels 450-499 - that is,
//         prod_k f'(x_k-y_k), where f' is the derivative of monotonic function
//         f for corresponding monotonic density kernel
//
// 600-649 monotonic density kernels of the form prod_k f(x_k-y_k), where f
//         is monotonic increasing, -1 < f(x) < 1, and f(0) = 0
// 650-699 monotonic density kernels of the form prod_k f(x_k-y_k), where f
//         is monotonic decreasing, -1 < f(x) < 1, and f(0) = 0
//
// 700-749 monotonic dense derivatives of kernels 600-649 - that is,
//         prod_k f'(x_k-y_k), where f' is the derivative of monotonic function
//         f for corresponding monotonic density kernel
// 750-799 monotonic dense derivatives of kernels 650-699 - that is,
//         prod_k f'(x_k-y_k), where f' is the derivative of monotonic function
//         f for corresponding monotonic density kernel
//
// 8xx use altcallback to evaluate kernel
//         f for corresponding monotonic density kernel
//
// 9xx get kernel evaluation from a server (i0 being a ref for kerni0.sock)




//
// Kernel Descriptions
// ===================
//
// rj = real constant j
// ij = integer constant j
// var(0,0) (x) = a = x'x
// var(0,1) (y) = b = y'y
// var(0,2) (z) = z = x'y
// var(0,3) = d = ||x-y||_2^2 = a+b-2*z
// (var(0,3) is substituted out for var(0,0)+var(0,1)-2*var(0,2) at end)
//
//KERNELSHERE - labels where kernel is actually evaluated
//
//- r0 should be lengthscale always but isn't for these kernels
//
// Number | Name                   | K(x,y)
// -------+------------------------+------------------------------
//     0  | Constant               | r1
//     1  | Linear                 | z/(r0.r0)
//     2  | Polynomial             | ( r1 + z/(r0.r0) )^i0
//     3  | Gaussian***            | exp(-d/(2.r0.r0)-r1)
//     4  | Laplacian***           | exp(-sqrt(d)/r0-r1)
//     5  | Polynoise***           | exp(-sqrt(d)^r1/(r1*r0^r1)-r2)
//     6  | ANOVA                  | sum_k exp(-r4*((x_k/r0)^r1-(y_k/r0)^r1)^r2)^r3
//     7  | Sigmoid#               | tanh( z/(r0.r0) + r1 )
//     8  | Rational quadratic     | ( 1 + d/(2*r0*r0*r1) )^(-r1)                         (was 1 - d/(d+r0))
//     9  | Multiquadratic%        | sqrt( d/(r0.r0) + r1^2 )
//    10  | Inverse multiquadric   | 1/sqrt( d/(r0.r0) + r1^2 )
//    11  | Circular*              | 2/pi * arccos(-sqrt(d)/r0) - 2/pi * sqrt(d)/r0 * sqrt(1 - d/r0^2)
//    12  | Sperical+              | 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3
//    13  | Wave                   | sinc(sqrt(d)/r0)
//    14  | Power                  | -sqrt(d/(r0.r0))^r1
//    15  | Log#                   | -log(sqrt(d/(r0.r0))^r1 + 1)
//    16  | Spline                 | prod_k ( 1 + (x_k/r0).(y_k/r0) + (x_k/r0).(y_k/r0).min(x_k/r0,y_k/r0) - ((x_k/r0+y_k/r0).min(x_k/r0,y_k/r0)^2)/2 + (min(x_k/r0,y_k/r0)^3)/3 )
//    17  | B-Spline               | sum_k B_(2i0+1)(x_k/r0-y_k/r0)
//    18  | Bessel^                | J_(i0+1) ( r1.sqrt(d)/r0) ) / ( (sqrt(d)/r0)^(-i0.(r1+1)) )
//    19  | Cauchy                 | 1/(1+(d/(r0.r0)))
//    20  | Chi-square             | 1 - sum_k (2((x_k/r0).(y_k/r0)))/(x_k/r0+y_k/r0)
//    21  | Histogram              | sum_k min(x_k/r0,y_k/r0)
//    22  | Generalised histogram  | sum_k min(|x_k/r0|^r1,|y_k/r0|^r2)
//    23  | Generalised T-student  | 1/(1+(sqrt(d)/r0)^r1)
//    24  | Vovk's real            | (1-((z/(r0.r0))^i0))/(1-(z/(r0.r0)))
//    25  | Weak fourier           | pi.cosh(pi-(sqrt(d)/r0))
//    26  | Thin spline 1          | ((d/r0)^(r1+0.5))
//    27  | Thin spline 2          | ((d/r0)^r1).ln(sqrt(d/r0))
//    28  | Generic                | (user defined)
//    29  | Arc-cosine~            | (1/pi) (r0.sqrt(a))^i0 (r0.sqrt(b))^i0 Jn(arccos(z/(sqrt(a).sqrt(b))))
//    30  | Chaotic logistic       | <phi_{sigma,n}(x/r0),phi_{sigma,n}(y/r0)>
//    31  | Summed chaotic logistic| sum_{0,n} Kn(x,y)
//    32  | Diagonal               | r1 if i == j >= 0, 0 otherwise
//    33  | Uniform                | 1/(2.r0) ( 1 if real(sqrt(d)) < r0, 0 otherwise )
//    34  | Triangular             | (1-sqrt(d)/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )
//    35  | Even-integer Matern    | ((2^(1-i0))/gamma(i0)).((sqrt(2.i0).sqrt(d)/r0)^i0).K_r1(sqrt(2.i0).sqrt(d)/r0)
//    36  | Weiner                 | prod_i min(x_i/r0,y_i/r0)
//    37  | Half-integer Matern    | exp(-(sqrt(2.(i0+1/2))/r0).sqrt(d)) . (gamma(i0+1)/gamma((2.i0)+1)) . sum_{i=0,1,...,i0}( ((i0+1)!/(i!.(i0-i)!)) . pow((sqrt(8.(i0+1/2))/r0).sqrt(d),i0-i) )
//    38  | 1/2-Matern             | exp(-sqrt(d)/r0)
//    39  | 3/2-Matern             | (1+((sqrt(3)/r0).sqrt(d))) . exp(-(sqrt(3)/r0).sqrt(d))
//    40  | 5/2-Matern             | (1+((sqrt(5)/r0).sqrt(d))+((5/(3.r0*r0))*d)) . exp(-(sqrt(5)/r0).sqrt(d))
//    41  | RBF-rescale            | exp(log(z)/(2.r0.r0))
//    42  | Inverse Gudermannian   | igd(z/(r0.r0))
//    43  | Log ratio              | log((1+z/(r0.r0))/(1-z/(r0.r0)))
//    44  | Exponential***         | exp(z/(r0.r0)-r1)
//    45  | Hyperbolic sine        | sinh(z/(r0.r0))
//    46  | Hyperbolic cosine      | cosh(z/(r0.r0))
//    47  | Sinc Kernel (Tobar)    | sinc(sqrt(d)/r0).cos(2*pi*sqrt(d)/(r0.r1))
//    48  | LUT kernel             | r1((int) x, (int) y) if r1 is a matrix, otherwise (r1 if x != y, 1 if x == y)
//        |                        |
//   100  | Linear 0/1             | z/(r0*r0)
//   101  | Logistic 0/1           | 1/(1+exp(-z/(r0*r0)))
//   102  | Generalised logstic 0/1| 1/(1+r1*exp(-r2*(z-r3)/(r0*r0)))^(1/r2)
//   103  | Heavyside 0/1          | 0 if real(z) < 0, 1 otherwise
//   104  | ReLU 0/1               | 0 if real(z) < 0, z/(r0*r0) otherwise
//   105  | Softplus 0/1           | ln(r1+exp(z/(r0*r0)))
//   106  | Leaky ReLU 0/1         | r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise
//        |                        |
//   200  | Linear -1/1            | z/(r0*r0)-1
//   201  | Logistic -1/1          | 2/(1+exp(-z/(r0*r0))) - 1
//   202  | Generalised logstc -1/1| 2/(1+r1*exp(-r2*(z-r3)/(r0*r0)))^(1/r2) - 1
//   203  | Heavyside -1/1         | -1 if real(z) < 0, 1 otherwise
//   204  | Relu -1/1              | -1 if real(z) < 0, z/(r0*r0)-1 otherwise
//   205  | Softplus -1/1          | 2.ln(r1+exp(z/(r0*r0))) - 1
//        |                        |
//   300  | Euclidean distance$    | -1/2 d/(r0.r0)
//   301  | 1-norm distance$       | -1/2 ||x-y||_1^2/(r0.r0)
//   302  | inf-norm distance$     | -1/2 ||x-y||_inf^2/(r0.r0)
//   303  | 0-norm distance$       | -1/2 ||x-y||_0^2/(r0.r0)
//   304  | r0-norm distance$      | -1/2 ||x-y||_real(r1)^2/(r0.r0)
//        |                        |
//   400  | Monotnic 0/1 dense 1   | (K600(x,y)+1)/2
//   401  | Monotnic 0/1 dense 2   | (K601(x,y)+1)/2
//   402  | Monotnic 0/1 dense 3   | (K602(x,y)+1)/2
//   403  | Monotnic 0/1 dense 4   | (K603(x,y)+1)/2
//   404  | Monotnic 0/1 dense 5   | (K604(x,y)+1)/2
//        |                        |
//   450  | Monotnic 0/1 dense 1rev| (K650(x,y)+1)/2
//   451  | Monotnic 0/1 dense 2rev| (K651(x,y)+1)/2
//   452  | Monotnic 0/1 dense 3rev| (K652(x,y)+1)/2
//   453  | Monotnic 0/1 dense 4rev| (K653(x,y)+1)/2
//   454  | Monotnic 0/1 dense 5rev| (K654(x,y)+1)/2
//        |                        |
//   500  | Monot dense deriv 1&   | K700(x,y)/2
//   501  | Monot dense deriv 2&   | K701(x,y)/2
//   502  | Monot dense deriv 3&'  | K702(x,y)/2
//   503  | Monot dense deriv 4&'  | K703(x,y)/2
//   504  | Monot dense deriv 5&'  | K704(x,y)/2
//        |                        |
//   550  | Monot dens deriv 1rev& | K750(x,y)/2
//   551  | Monot dens deriv 2rev& | K751(x,y)/2
//   552  | Monot dens deriv 3rev&`| K752(x,y)/2
//   553  | Monot dens deriv 5rev&`| K753(x,y)/2
//   554  | Monot dens deriv 5rev&`| K754(x,y)/2
//        |                        |
//   600  | Monot. -1/+1 density 1 | prod_k ( 2/(1+exp(-(x_k-y_k)/r0)) - r1 )
//   601  | Monot. -1/+1 density 2 | prod_k ( erf((x_k-y_k)/r0 - r1 )
//   602  | Monot. -1/+1 density 3 | 2/(1+exp(-min_k(x_k-y_k)/r0)) - r1
//   603  | Monot. -1/+1 density 4 | -1 if real(min_k(x_k-y_k)) < 0, 1 otherwise
//   604  | Monot. -1/+1 density 5 | max_k(x_k-y_k)/r0
//        |                        |
//   650  | Monot. -1/+1 dense 1rev| K600(y,x)
//   651  | Monot. -1/+1 dense 2rev| K601(y,x)
//   652  | Monot. -1/+1 density 3 | K602(y,x)
//   653  | Monot. -1/+1 density 4 | K603(y,x)
//   654  | Monot. -1/+1 density 5 | K604(y,x)
//        |                        |
//   700  | Mon -1+1 dens deriv 1& | prod_k ( (2/r0).exp(-(x_k-y_k)/r0)/(1+exp(-(x_k-y_k)/r0))^2 )
//   701  | Mon -1+1 dens deriv 2& | prod_k ((2/r0)/sqrt(pi))*exp(-((x_k-y_k)/r0)^2)
//   702  | Mon -1+1 dens deriv 3&'| (2/r0).exp(-min_k(x_k-y_k)/r0)/((1+exp(-max_k(x_k-y_k)/r0))^2)
//   703  | Mon -1+1 dens deriv 4&'| 0
//   704  | Mon -1+1 dens deriv 5&'| 1/r0
//        |                        |
//   750  | Mon dens+- deriv 1rev& | -K700(y,x)
//   751  | Mon dens+- deriv 2rev& | -K701(y,x)
//   752  | Mon dens+- deriv 3rev&`| -K702(y,x)
//   753  | Mon dens+- deriv 5rev&`| -K703(y,x)
//   754  | Mon dens+- deriv 5rev&`| -K704(y,x)
//        |                        |
//   8xx  | altcallback kernel eval| Uses altcallback to evaluate kernel.  Assumed symmetric
//        |                        |
//   9xx  | stream kernel evaluatio| Evaluate kernel via stream.  Uses UNIX socket kern_.sock, where _ is i0.
//        |                        | Client is opened when required, server is open now.
//        |                        | Mercer sends data and vectors (see kernel9xx function) and waits for
//        |                        | client to return the result.
//        |                        | 900: unix socket server symmetric
//        |                        | 901: unix socket server anti-symmetric
//        |                        | 902: unix socket server asymmetric
//        |                        | 903: unix socket client symmetric
//        |                        | 904: unix socket client anti-symmetric
//        |                        | 905: unix socket client asymmetric
//
// Notes: % non-mercer
//        * only positive definite in R^2
//        + only positive definite in R^3
//        # conditionally positive definite
//        ^ not yet implemented
//        ~ see Youngmin Cho, Lawrence K. Saul - Kernel Methods for Deep
//          Learning
//        $ note that K(x,x) + K(y,y) - 2.K(x,y) = ||x-y||_q^2 (q is relevant
//          norm)
//        & These are kernels 4xx with d/dx0 d/dx1 ... applied
//        ` See design decision in dense derivative
//        @ These are kernels 6xx with d/dx0 d/dx1 ... applied
//
// where: - B_i(z) = (1/i!) sum_{j=0 to i+1} (i+1)choose(j) (-1)^j max(0,(z + (i+1)/2 - j))^i
//        - Jn(x) = sin^(2n+1) (-1/sin(x) d/dx)^n (pi-x)/sin(x)
//        - phi_{sigma,n}(x) = phi_sigma(phi_sigma(...phi_sigma(x)))
//          (n repeats)
//        - phi_sigma(x) = ( sigma.x_0.(1-x_0) sigma.x_1.(1-x_1) ... )
//        - Kn is the Chaotic Logistic Kernel (case 30)
//        - K_r0 is the modified Bessel function
//        - If r1 = 3/2 the Matern kernel is
//          K(x,y) = ( 1 + sqrt(3).||x-y||/r0 ) exp(-sqrt(3).||x-y||/r0)
//          If r1 = 5/2 the Matern kernel is
//          K(x,y) = ( 1 + sqrt(5).||x-y||/r0 + 5.||x-y||^2/(3.r0^2) ) exp(-sqrt(5).||x-y||/r0)
//
// Generic kernel:
//
// Treating r10 as a function (which it is), evaluate:
//
// K(x,y) = r10(varxy)
//
// where varxy is:
//
// varxy(0,0) = m
// varxy(0,1) = x'y
// varxy(0,2) = y'x
// varxy(0,3) = (x-y)'(x-y)
// varxy(1,i) = ri (as is, not evaluated in any way, including r10)
// varxy(2,i) = Ki(x,y)  (evaluated on an is-used basis)
// varxy(3,.) = x
// varxy(4,.) = y
//
// Note that varxy(2,i) must be referenced directly or it will not
// be evaluated a-priori, leading to the wrong result.  So for example
// sum_i var(2,i) will not work.
//
// Note also that only x and y are available here


//
// Real constant derivatives
// =========================
//
//KERNELSHERE - labels where kernel is actually evaluated
//
// ... means calculate on the fly (if possible)
//
// Number | Name                   | K(x,y)
// -------+------------------------+------------------------------
//     0  | Constant               | ( 0 )
//        |                        | ( 1 )
//     1  | Linear                 | ( -2.z/(r0.r0.r0))
//     2  | Polynomial             | ( -2.i0.(x'y)/(r0.r0.r0) * ( r1 + x'y/(r0.r0) )^(i0-1) )
//        |                        | (    i0.                 * ( r1 + x'y/(r0.r0) )^(i0-1) )
//     3  | Gaussian               | ( (d/(r0*r0*r0)).exp(-d/(2*r0*r0)-r1) )
//     4  | Laplacian              | ( (sqrt(d)/(r0*r0)).exp(sqrt(d)/r0-r1) )
//     5  | Polynoise              | ( (sqrt(d)^r1)/(r0^(r1+1))                                                           exp(-sqrt(d)^r1/(r1*r0^r1)-r2) )
//        |                        | ( ( ((sqrt(d)^r1)/((r1^2).(r0^r1))) - (log(sqrt(d)/r0)/r1).exp(r1.log(sqrt(d)/r0)) ) exp(-sqrt(d)^r1/(r1*r0^r1)-r2) )
//     6  | ANOVA                  | ...
//     7  | Sigmoid                | ( -2.z/(r0.r0.r0) sech^2 ( z/(r0.r0) + r1 ) )
//        |                        | (                 sech^2 ( z/(r0.r0) + r1 ) )
//     8  | Rational quadratic     | ...                   (was ( d/((d+r0)^2) ))
//     9  | Multiquadratic         | ( -(2.d/(r0.r0.r0))/sqrt(d/(r0.r0)+r1^2) )
//        |                        | (                r1/sqrt(d/(r0.r0)+r1^2) )
//    10  | Inverse multiquadric   | ( (2.d/(r0.r0.r0))/(sqrt(d/(r0.r0)+r1^2))^3 )
//        |                        | (              -r1/(sqrt(d/(r0.r0)+r1^2))^3 )
//    11  | Circular               | ( -4/pi  (||x-y||^3/r0^3)/sqrt(1-(d/r0^2))  1/r0 )
//    12  | Sperical               | ( 3/2 (1-(d/r0^2)) sqrt(d)/r0^2 )
//    13  | Wave                   | ( -1/r0 ( cos(d/r0) - sinc(d/r0) ) )
//    14  | Power                  | ...
//    15  | Log                    | ...
//    16  | Spline                 | ...
//    17  | B-Spline               | ...
//    18  | Bessel                 | Not currently implemented
//    19  | Cauchy                 | ...
//    20  | Chi-square             | ...
//    21  | Histogram              | ...
//    22  | Generalised histogram  | ...
//    23  | Generalised T-student  | ...
//    24  | Vovk's real            | ...
//    25  | Weak fourier           | ...
//    26  | Thin spline 1          | ...
//    27  | Thin spline 2          | ...
//    28  | Generic                | ...
//    29  | Arc-cosine             | ...
//    30  | Chaotic logistic       | ...
//    31  | Summed chaotic logistic| ()
//    32  | Diagonal               |
//    33  | Uniform                | ...
//    34  | Triangular             | ...
//    35  | Matern                 | Not currently implemented
//    36  | Weiner                 | (-K/(r0.r0))
//    37  | Half-integer Matern    | ...
//    38  | 1/2-Matern             | ...
//    39  | 3/2-Matern             | ...
//    40  | 5/2-Matern             | ...
//    41  | RBF-rescale            | ...
//    42  | Inverse Gudermannian   | ...
//    43  | Log ratio              | ...
//    44  | Exponential            | ...
//    45  | Hyperbolic sine        | ...
//    46  | Hyperbolic cosine      | ...
//    47  | Sinc Kernel (Tobar)    | ...
//    48  | LUT kernel             | ...
//        |                        |
//   100  | Linear 0/1             | ...
//   101  | Logistic 0/1           | ...
//   102  | Generalised logstic 0/1| ...
//   103  | Heavyside 0/1          | ( 0 )
//   104  | Rectifier 0/1          | ...
//   105  | Softplus 0/1           | ...
//   106  | Leaky rectifier 0/1    | ...
//        |                        |
//   200  | Linear -1/1            | ...
//   201  | Logistic -1/1          | ...
//   202  | Generalised logstc -1/1| ...
//   203  | Heavyside -1/1         | ( 0 )
//   204  | Rectifier -1/1         | ...
//   205  | Softplus -1/1          | ...
//        |                        |
//   300  | Euclidean distance$    | ...
//   301  | 1-norm distance$       | ...
//   302  | inf-norm distance$     | ...
//   304  | 0-norm distance$       | ...
//   305  | r0-norm distance$      | ...
//        |                        |
//   400  | Monotnic 0/1 dense 1   | ...
//   401  | Monotnic 0/1 dense 2   | ...
//   402  | Monotnic 0/1 dense 3   | ...
//   403  | Monotnic 0/1 dense 4   | ...
//   404  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   450  | Monotnic 0/1 dense 1rev| ...
//   451  | Monotnic 0/1 dense 2rev| ...
//   452  | Monotnic 0/1 dense 3rev| ...
//   453  | Monotnic 0/1 dense 4rev| ...
//   454  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   500  | Monot dense deriv 1&   | ...
//   501  | Monot dense deriv 2&   | ...
//   502  | Monot dense deriv 3&`  | ...
//   503  | Monot dense deriv 4&`  | ...
//   504  | Monot dense deriv 5&`  | ...
//        |                        |
//   550  | Monot dens deriv 1rev&`| ...
//   551  | Monot dens deriv 2rev&`| ...
//   552  | Monot dens deriv 3rev&`| ...
//   553  | Monot dens deriv 4rev&`| ...
//   554  | Monot dens deriv 5rev&`| ...
//        |                        |
//   600  | Monotnic 0/1 dense 1   | ...
//   601  | Monotnic 0/1 dense 2   | ...
//   602  | Monotnic 0/1 dense 3   | ...
//   603  | Monotnic 0/1 dense 4   | ...
//   604  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   650  | Monotnic 0/1 dense 1rev| ...
//   651  | Monotnic 0/1 dense 2rev| ...
//   652  | Monotnic 0/1 dense 3rev| ...
//   653  | Monotnic 0/1 dense 4rev| ...
//   654  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   700  | Monot dense deriv 1&   | ...
//   701  | Monot dense deriv 2&   | ...
//   702  | Monot dense deriv 3&`  | ...
//   703  | Monot dense deriv 4&`  | ...
//   704  | Monot dense deriv 5&`  | ...
//        |                        |
//   750  | Monot dens deriv 1rev&`| ...
//   751  | Monot dens deriv 2rev&`| ...
//   752  | Monot dens deriv 3rev&`| ...
//   753  | Monot dens deriv 4rev&`| ...
//   754  | Monot dens deriv 5rev&`| ...
//        |                        |
//   800  | altcallback kernel eval| ()
//        |                        |
//   900  | altcallback kernel eval| not defined
//


//
// Kernels derivatives
// ===================
//
//KERNELSHERE - labels where kernel is actually evaluated
//
// Number | Name                   | dK(x,y)/dz
// -------+------------------------+------------------------------
//     0  | Constant               | 0
//     1  | Linear                 | 1/(r0*r0)
//     2  | Polynomial             | i0/(r0.r0) * ( r1 + z/(r0.r0) )^(i0-1)
//     3  | Gaussian               | K(x,y)/(r0*r0)
//     4  | Laplacian              | K(x,y)/(r0*sqrt(d))
//        |                        | (arbitrarily 1 if x == y in line with RBF)
//     5  | Polynoise              | K(x,y) * ((sqrt(d)^(r1-2))/(r0^r1))
//        |                        | (arbitrarily 1 if x == y in line with RBF)
//     6  | ANOVA                  | ...
//     7  | Sigmoid                | 1/(r0.r0) * sech^2( z/(r0.r0) + r1 )
//     8  | Rational quadratic     | -(1/(2*r0*r0)).( 1 + d/(2*r0*r0*r1) )^(-r1-1)             was 2.r0/((d+r0)^2)
//     9  | Multiquadratic%        | -(1/(r0.r0))/K(x,y)
//    10  | Inverse multiquadric   | (1/(r0.r0)).K(x,y)^3
//    11  | Circular               | -4/(pi*r0^2) ( sqrt(diffis/(r0^2-diffis)) )
//    12  | Sperical               | 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3
//    13  | Wave                   | -2 (cos(sqrt(d)/r0) - sinc(sqrt(d)/r0))/sqrt(d)/r0 1/(2*r0^2*sqrt(d)/r0)
//    14  | Power                  | (r1 * (d/(r0.r0))^((r1/2)-1))/(r0.r0)
//    15  | Log#                   | (r1 * ((d/(r0.r0))^((r1/2)-1))/d^(r0/2) + 1))/(r0.r0)
//    16  | Spline                 | ...
//    17  | B-Spline               | ...
//    18  | Bessel^                | Not currently implemented
//    19  | Cauchy                 | 2/(r0.r0) * K(x,y)^2
//    20  | Chi-square             | ...
//    21  | Histogram              | ...
//    22  | Generalised histogram  | ...
//    23  | Generalised T-student  | ...
//    24  | Vovk's real            | ( ( -i0.((z/(r0.r0))^(i0-1)) + (1-((z/(r0.r0))^i0))/(1-(z/(r0.r0))) )/(1-(z/(r0.r0))) )/(r0.r0)
//        |                        | (ill-defined at z = 1)
//    25  | Weak fourier           | pi/r0 * sinh(pi-sqrt(d)/r0) / sqrt(d)
//    26  | Thin spline 1          | -2/r0 * (r1+0.5) * (d/r0)^(r1-0.5)
//    27  | Thin spline 2          | -(2.r1)/r0 * ( (d/r0)^r1 * ln(sqrt(d/r0)) + 1/2 ) / (d/r0)
//    28  | Generic                | ...
//    29  | Arc-cosine*            | ...
//    30  | Chaotic logistic       | ...
//    31  | Summed chaotic logistic| ...
//    32  | Diagonal               | ...
//    33  | Uniform                | 0.0
//    34  | Triangular             | (1-real(sqrt(d))/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )
//    35  | Matern^                | Not currently implemented
//    36  | Weiner                 | ...
//    37  | Half-integer Matern    | Gradient not currently implemented
//    38  | 1/2-Matern             | Gradient not currently implemented
//    39  | 3/2-Matern             | Gradient not currently implemented
//    40  | 5/2-Matern             | Gradient not currently implemented
//    41  | RBF-rescale            | Gradient not currently implemented
//    42  | Inverse Gudermannian   | (1/(r0.r0)) sec^2(z/(r0.r0)) / ( 1 - tan^2(z/(r0.r0)) )
//    43  | Log ratio              | -(1/(r0.r0))^2/(1-z/(r0.r0))^2
//    44  | Exponential            | exp(z/(r0.r0)-r1)/(r0.r0)
//    45  | Hyperbolic sine        | sinh(z/(r0.r0))/(r0.r0)
//    46  | Hyperbolic cosine      | cosh(z/(r0.r0))/(r0.r0)
//    47  | Sinc Kernel (Tobar)    | ...
//    48  | LUT kernel             | ...
//        |                        |
//   100  | Linear 0/1             | 1
//   101  | Logistic 0/1           | (K(x,y).(1-K(x,y)))/(r0*r0)
//   102  | Generalised logstic 0/1| (K(x,y).(1-K(x,y)^r2))/(r0*r0)
//   103  | Heavyside 0/1          | 0.0
//   104  | Rectifier 0/1          | 0 if real(z) < 0, 1/(r0*r0) otherwise
//   105  | Softplus 0/1           | (exp(r0*z)/(r1+exp(r0*z)))/(r0*r0)
//   106  | Leaky Rectifier 0/1    | r1/(r0*r0) if real(z) < 0, 1/(r0*r0) otherwise
//        |                        |
//   200  | Linear -1/1            | 1
//   201  | Logistic -1/1          | ((1-K(x,y)).(1+K(x,y))/2)/(r0*r0)
//   202  | Generalised logstc -1/1| 2.((1-K102(x,y)^r2).K102(x,y))/(r0*r0)
//   203  | Heavyside -1/1         | 0
//   204  | Rectifier -1/1         | 0 if real(z) < 0, 1/(r0*r0) otherwise
//   205  | Softplus -1/1          | 2.(exp(r0.z)/(r1+exp(r0.z)))/(r0*r0)
//        |                        |
//   300  | Euclidean distance     | ...
//   301  | 1-norm distance        | ...
//   302  | inf-norm distance      | ...
//   304  | 0-norm distance        | ...
//   305  | r0-norm distance       | ...
//        |                        |
//   400  | Monotnic 0/1 dense 1   | ...
//   401  | Monotnic 0/1 dense 2   | ...
//   402  | Monotnic 0/1 dense 3   | ...
//   403  | Monotnic 0/1 dense 4   | ...
//   404  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   450  | Monotnic 0/1 dense 1rev| ...
//   451  | Monotnic 0/1 dense 2rev| ...
//   452  | Monotnic 0/1 dense 3rev| ...
//   453  | Monotnic 0/1 dense 4rev| ...
//   454  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   500  | Monot dense deriv 1    | ...
//   501  | Monot dense deriv 2    | ...
//   502  | Monot dense deriv 3    | ...
//   503  | Monot dense deriv 4    | 0
//   504  | Monot dense deriv 5    | 0
//        |                        |
//   550  | Monot dense deriv 1    | ...
//   551  | Monot dense deriv 2    | ...
//   552  | Monot dens deriv 3rev  | ...
//   553  | Monot dens deriv 4rev  | 0
//   554  | Monot dens deriv 5rev  | 0
//        |                        |
//   600  | Monotnic 0/1 dense 1   | ...
//   601  | Monotnic 0/1 dense 2   | ...
//   602  | Monotnic 0/1 dense 3   | ...
//   603  | Monotnic 0/1 dense 4   | ...
//   604  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   650  | Monotnic 0/1 dense 1rev| ...
//   651  | Monotnic 0/1 dense 2rev| ...
//   652  | Monotnic 0/1 dense 3rev| ...
//   653  | Monotnic 0/1 dense 4rev| ...
//   654  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   700  | Monot dense deriv 1    | ...
//   701  | Monot dense deriv 2    | ...
//   702  | Monot dense deriv 3    | ...
//   703  | Monot dense deriv 4    | 0
//   704  | Monot dense deriv 5    | 0
//        |                        |
//   750  | Monot dense deriv 1    | ...
//   751  | Monot dense deriv 2    | ...
//   752  | Monot dens deriv 3rev  | ...
//   753  | Monot dens deriv 4rev  | 0
//   754  | Monot dens deriv 5rev  | 0
//        |                        |
//   800  | altcallback kernel eval| not defined
//        |                        |
//   900  | altcallback kernel eval| not defined

// Notes: * The derivative of the arc-cosine kernel is not implemented
//          (although the arc-cosine kernel itself is).
//        ^ not yet implemented
//
//
// Working:
// ========
//
// OLD Case 8: Rational quadratic kernel:
// OLD
// OLD K(x,y) = 1 - ||x-y||^2/(||x-y||^2+r0)
// OLD        = 1 - ( a + b - 2z )/(( a + b - 2z )+r0)
// OLD
// OLD dK/dz = 2/((a+b-2z)+r0)  -  2.(a+b-2z)/(((a+b-2z)+r0)*((a+b-2z)+r0))
// OLD       = 2.( ((a+b-2z)+r0) - (a+b-2z) )/(((a+b-2z)+r0)*((a+b-2z)+r0))
// OLD       = 2.r0/(((a+b-2z)+r0)*((a+b-2z)+r0))
// OLD       = 2.r0/((diffis+r0)*(diffis+r0))
//
// Case 11: Circular kernel (ONLY POS DEFINITE IN R^2):
//
// K(x,y) = 2/pi * arccos(-||x-y||/r0) - 2/pi * ||x-y||/r0 * sqrt(1 - ||x-y||^2/r0^2)
//        = 2/pi * arccos(-sqrt(a+b-2z)/r0) - 2/pi * sqrt(a+b-2z)/r0 * sqrt(1 - (a+b-2z)/r0^2)
//
// q = diffis/r0^2
//
// K(diffis) = 2/pi * arccos(-sqrt(q))
//           - 2/pi * sqrt(q) * sqrt(1-q)
//
//           = 2/pi * ( arccos(-sqrt(q))
//                    - sqrt(q) * sqrt(1-q) )
//
// dK/dq = 2/pi * ( -1/2 * 1/sqrt(q) * -1/sqrt(1-q)
//                - 1/2 * 1/sqrt(q) * sqrt(1-q)
//                - sqrt(q) * -1/2 * 1/sqrt(1-q) )
//
//            = 2/pi * ( 1/2 * 1/sqrt(q) * 1/sqrt(1-q)
//                     - 1/2 * 1/sqrt(q) * sqrt(1-q)
//                     + 1/2 * sqrt(q) * 1/sqrt(1-q) )
//
//            = 1/pi * (       * 1/sqrt(q) * 1/sqrt(1-q)
//                     - (1-q) * 1/sqrt(q) * 1/sqrt(1-q)
//                     + q     * 1/sqrt(q) * 1/sqrt(1-q) )
//
//            = 2/pi * ( q * 1/sqrt(q) * 1/sqrt(1-q) )
//
//            = 2/pi * ( sqrt(q)/sqrt(1-q) )
//
// dK/dz = dK/dq * dq/ddiffis * ddiffis/dz
//       = dK/dq * 1/r0^2 * -2
//       = -2/r0^2 dK/dq
//       = -4/(pi*r0^2) ( sqrt(q)/sqrt(1-q) )
//       = -4/(pi*r0^2) ( sqrt(diffis)/sqrt(r0^2-diffis) )
//       = -4/(pi*r0^2) ( sqrt(diffis/(r0^2-diffis)) )
//
// Case 12: Spherical kernel (ONLY POS DEFINITE IN R^3):
//
// K(x,y) = 1 - 3/2 * ||x-y||/r0 + 1/2 * ||x-y||^3/r0^3
//        = 1 - 3/2 * sqrt( x'x + y'y - 2x'y )/r0 + 1/2 * ( x'x + y'y - 2x'y )^(3/2)/r0^3
//
// K(diffis) = 1 - 3/2 * sqrt(diffis)/r0 + 1/2 * sqrt(diffis)^3/r0^3
//
// dK/ddiffis = -3/2 1/2 1/sqrt(diffis) 1/r0 + 1/2 3/2 sqrt(diffis) 1/r0^3
// dK/ddiffis = -3/(4*r0) 1/sqrt(diffis) + 3/(4*r0^3) sqrt(diffis)
// dK/dz = -2 dK/ddiffis
//       = 3/(2*r0) 1/sqrt(diffis) - 3/(2*r0^3) sqrt(diffis)
//       = 3/(2*r0^3) ( r0*r0/sqrt(diffis) - sqrt(diffis) )
//
// Case 13: Wave kernel:
//
// K(x,y) = (r0/||x-y||).sin(||x-y||/r0)
//        = sinc(||x-y||/r0)
//        = (r0/sqrt( x'x + y'y - 2x'y )).sin(sqrt( x'x + y'y - 2x'y )/r0)
//
// K(diffis) = (r0/sqrt(diffis)).sin(sqrt(diffis)/r0)
//
// q = sqrt(diffis)/r0
//
// K(q) = sin(q)/q
//
// dK/dq = cos(q)/q - sin(q)/q^2
//       = (cos(q) - sin(q)/q)/q
//       = (cos(q) - sinc(q))/q
//
// dq/ddiffis = 1/(2*r0) * 1/sqrt(diffis)
//            = 1/(2*r0^2) * r0/sqrt(diffis)
//            = 1/(2*r0^2*q)
//
// ddiffis/dz = -2
//
// dK/dz = dK/dq dq/ddiffis ddiffis/dz
//       = -2 (cos(sqrt(diffis)/r0) - sinc(sqrt(diffis)/r0))/sqrt(diffis)/r0 1/(2*r0^2*sqrt(diffis)/r0)
//
// Case 19: Cauchy kernel:
//
// K(x,y) = 1/(1+((||x-y||^2)/r0))
//        = 1/(1+(( x'x + y'y - 2x'y )/r0))
//        = 1/(1 + ((a+b-2z)/r0) )
//
// dK/dz = d/dz 1/(1 + ((a+b-2z)/r0) )
//       = 2/r0 * 1/(1 + ((a+b-2z)/r0) )^2
//       = 2/r0 * K(diffis)^2
//
// Case 23: Generalised T-Student kernel:
//
// K(x,y) = 1/(1+(||x-y||/r0)^r1)
//        = 1/(1+(( x'x + y'y - 2x'y )/r0)^r1)
//        = 1/(1+ ((a+b-2z)/r0)^r1 )
//
// dK/dz = (2.r1/r0) * ((a+b-2z)/r0)^(r1-1)/(1+ ((a+b-2z)/r0)^r1 )^2
//       = (2.r1/r0) * diffis^(r1-1) * K(diffis)^2
//
// Case 24: Vovk's real polynomial:
//
// K(x,y) = (1-((x'y)^i0))/(1-(x'y))
//        = (1-(z^i0))/(1-z)
// (0 as z->1)
//
// dK/dz =  -i0.(z^(i0-1))/(1-z) + (1-(z^i0))/((1-z)^2)
// dK/dz =  ( -i0.(z^(i0-1)) + (1-(z^i0))/(1-z) )/(1-z)
// (ill-defined as z->1, so don't try)
//
// Case 25: Weak fourier kernel:
//
// K(x,y) = pi.cosh(pi-(||x-y||/r0))
//        = pi.cosh(pi-(sqrt(x'x + y'y - 2x'y)/r0))
//
// K(diffis) = pi.cosh(pi-sqrt(diffis)/r0)
//
// dK/ddiffis = pi * sinh(pi-sqrt(diffis)/r0) * -1/r0 * 1/2 * 1/sqrt(diffis)
//            = -pi/(2*r0) * sinh(pi-sqrt(diffis)/r0) / sqrt(diffis)
//
// dK/dz  = dK/ddiffis ddiffis/dz = -2 dK/ddiffis
//        = pi/r0 * sinh(pi-sqrt(diffis)/r0) / sqrt(diffis)
//
// Case 26: Thin spline (1):
//
// K(x,y) = ((||x-y||^2/r0)^(r1+0.5))
//        = (((x'x + y'y - 2x'y)/r0)^(r1+0.5))
//
// K(diffis) = (diffis/r0)^(r1+0.5)
//
// dK/ddiffis = 1/r0 * (r1+0.5) * (diffis/r0)^(r1-0.5)
//
// Case 27: Thin spline (2):
//
// K(x,y) = ((||x-y||^2/r0)^r1).ln(sqrt(||x-y||^2/r0))
//        = (((x'x + y'y - 2x'y)/r0)^r1).ln(sqrt((x'x + y'y - 2x'y)/r0))
//
// q = diffis/r0
//
// K(q) = (q^r1).ln(sqrt(q))
//
// dK/dq = r1 * q^(r1-1) * ln(sqrt(q))
//       + r1 * 1/sqrt(q) * 1/2 * 1/sqrt(q)
//       = r1 * ( q^(r1-1) * ln(sqrt(q)) + 1/2q )
//       = r1 * ( q^r1 * ln(sqrt(q)) + 1/2 ) / q
//
// dK/dz = dK/dq * dq/ddiffis * ddiffis/dz
//       = dK/dq * 1/r0 * -2
//       = -2/r0 dK/dq
//       = -(2.r1)/r0 * ( (d/r0)^r1 * ln(sqrt(d/r0)) + 1/2 ) / (d/r0)
//
// Case 29: Arccosine:
//
// K(x,y) = (1/pi) r0^2 ||x||^i0 ||y||^i0 Jn(arccos(x'y/(||x||.||y||)))
//        = (1/pi) r0^2 sqrt(a,b)^i0 Jn(arccos(z/sqrt(a.b)))
//
// dK/dz = (1/pi) r0^2 sqrt(a.b)^i0 dJn/dtheta(arccos(z/sqrt(a.b))) 1/sqrt(1-(z^2/(a.b))) 1/sqrt(a.b)
//       = (1/pi) r0^2 sqrt(a.b)^(i0-1) dJn/dtheta(arccos(z/sqrt(a.b))) 1/sqrt(1-(z^2/(a.b)))
//
// let q = z/sqrt(a.b)
//
// dK/dz = (1/pi) r0^2 sqrt(a.b)^(i0-1)/sqrt(1-q^2) dJn/dtheta(arccos(q))
//
// dK/da = (1/pi) r0^2 sqrt(b)^i0 Jn(arccos(z/sqrt(a.b)))
//
//FIXME: finish this derivation, implement it
//
// Case 34: Triangular kernel
//
// K(x,y) = (1-||x-y||/r0)/r0 if real(||x-y||) < r0, 0 otherwise )
//        = (1-sqrt(d)/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )
//
// dK/dz = -2 dK/dd
//       = -2 -1/2r0 1/sqrt(d) 1/r0 if real(sqrt(d)) < r0, 0 otherwise
//       = 1/r0^2 1/sqrt(d) if real(sqrt(d)) < r0, 0 otherwise
//
// Case 501:
//
// K(d) = ((1.r0/sqrt(pi))^k)*exp(-r0*d)
//
// dK/dd = -r0.((1.r0/sqrt(pi))^k)*exp(-r0*d)
// dK/dz = -2 dK/dd = 2.r0.K(x,y)
//
// Case 701:
//
// K(d) = ((2.r0/sqrt(pi))^k)*exp(-r0*d)
//
// dK/dz = 2.r0.K(x,y)

// Second-order kernel derivatives
//
// q = a+b-2z
//
// Let e,f = a,b,z
//
// d2K/dede = d/de ( dK/de )
//          = dq/de d/dq ( dq/de dK/dq )
//          = dq/de ( de/dq d/de dq/de ) dK/dq + dq/de kq/de d2K/dqdq
//          = dq/de de/dq d2q/dede dK/dq + dq/de dq/de d2K/dqdq
//          = dq/de dq/de d2K/dqdq
//          = dz/de dz/de d2K/dzdz
//
// d2K/dedf = d/de ( dK/df )
//          = dq/de d/dq ( dq/df dK/dq )
//          = dq/de ( df/dq d/df dq/df ) dK/dq + dq/de dq/df d2K/dqdq
//          = dq/de df/dq d2q/dfdf dK/dq + dq/de dq/df d2K/dqdq
//          = dq/de dq/df d2K/dqdq
//          = dz/de dz/df d2K/dzdz
//
// So: need only /dqdq and /dzdz variants, rest can be calculated from that
//
//KERNELSHERE - labels where kernel is actually evaluated
//
// Number | Name                   | d2K(x,y)/dz2
// -------+------------------------+------------------------------
//     0  | Constant               | 0
//     1  | Linear                 | 0
//     2  | Polynomial             | i0.(i0-1)/(r0.r0.r0.r0) * ( r1 + z/(r0.r0) )^(i0-2)
//     3  | Gaussian               | K(x,y)/(r0*r0*r0*r0)
//     4  | Laplacian              | Not currently implemented
//     5  | Polynoise              | Not currently implemented
//     6  | ANOVA                  | ...
//     7  | Sigmoid                | Not currently implemented
//     8  | Rational quadratic     | Not currently implemented
//     9  | Multiquadratic%        | Not currently implemented
//    10  | Inverse multiquadric   | Not currently implemented
//    11  | Circular               | Not currently implemented
//    12  | Sperical               | Not currently implemented
//    13  | Wave                   | Not currently implemented
//    14  | Power                  | Not currently implemented
//    15  | Log#                   | Not currently implemented
//    16  | Spline                 | ...
//    17  | B-Spline               | ...
//    18  | Bessel^                | Not currently implemented
//    19  | Cauchy                 | Not currently implemented
//    20  | Chi-square             | ...
//    21  | Histogram              | ...
//    22  | Generalised histogram  | ...
//    23  | Generalised T-student  | Not currently implemented
//    24  | Vovk's real            | Not currently implemented
//    25  | Weak fourier           | Not currently implemented
//    26  | Thin spline 1          | Not currently implemented
//    27  | Thin spline 2          | Not currently implemented
//    28  | Generic                | ...
//    29  | Arc-cosine*            | ...
//    30  | Chaotic logistic       | ...
//    31  | Summed chaotic logistic| ...
//    32  | Diagonal               | 0
//    33  | Uniform                | 0
//    34  | Triangular             | Not currently implemented
//    35  | Matern^                | Not currently implemented
//    36  | Weiner                 | ...
//        |                        |
//   100  | Linear 0/1             | 0
//   101  | Logistic 0/1           | Not currently implemented
//   102  | Generalised logstic 0/1| Not currently implemented
//   103  | Heavyside 0/1          | 0
//   104  | Rectifier 0/1          | 0
//   105  | Softplus 0/1           | Not currently implemented
//   106  | Leaky rectifier 0/1    | 0
//        |                        |
//   200  | Linear -1/1            | 0
//   201  | Logistic -1/1          | Not currently implemented
//   202  | Generalised logstc -1/1| Not currently implemented
//   203  | Heavyside -1/1         | 0
//   204  | Rectifier -1/1         | 0
//   205  | Softplus -1/1          | Not currently implemented
//        |                        |
//   300  | Euclidean distance     | ...
//   301  | 1-norm distance        | ...
//   302  | inf-norm distance      | ...
//   304  | 0-norm distance        | ...
//   305  | r0-norm distance       | ...
//        |                        |
//   400  | Monotnic 0/1 dense 1   | ...
//   401  | Monotnic 0/1 dense 2   | ...
//   402  | Monotnic 0/1 dense 3   | ...
//   403  | Monotnic 0/1 dense 4   | ...
//   404  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   450  | Monotnic 0/1 dense 1rev| ...
//   451  | Monotnic 0/1 dense 2rev| ...
//   452  | Monotnic 0/1 dense 3rev| ...
//   453  | Monotnic 0/1 dense 4rev| ...
//   454  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   500  | Monot dense deriv 1    | ...
//   501  | Monot dense deriv 2    | ...
//   502  | Monot dense deriv 3    | ...
//   503  | Monot dense deriv 4    | 0
//   504  | Monot dense deriv 5    | 0
//        |                        |
//   550  | Monot dense deriv 1    | ...
//   551  | Monot dense deriv 2    | ...
//   552  | Monot dens deriv 3rev  | ...
//   553  | Monot dens deriv 4rev  | 0
//   554  | Monot dens deriv 5rev  | 0
//        |                        |
//   600  | Monotnic 0/1 dense 1   | ...
//   601  | Monotnic 0/1 dense 2   | ...
//   602  | Monotnic 0/1 dense 3   | ...
//   603  | Monotnic 0/1 dense 4   | ...
//   604  | Monotnic 0/1 dense 5   | ...
//        |                        |
//   650  | Monotnic 0/1 dense 1rev| ...
//   651  | Monotnic 0/1 dense 2rev| ...
//   652  | Monotnic 0/1 dense 3rev| ...
//   653  | Monotnic 0/1 dense 4rev| ...
//   654  | Monotnic 0/1 dense 5rev| ...
//        |                        |
//   700  | Monot dense deriv 1    | ...
//   701  | Monot dense deriv 2    | ...
//   702  | Monot dense deriv 3    | ...
//   703  | Monot dense deriv 4    | 0
//   704  | Monot dense deriv 5    | 0
//        |                        |
//   750  | Monot dense deriv 1    | ...
//   751  | Monot dense deriv 2    | ...
//   752  | Monot dens deriv 3rev  | ...
//   753  | Monot dens deriv 4rev  | 0
//   754  | Monot dens deriv 5rev  | 0
//        |                        |
//   800  | altcallback kernel eval| not defined
//        |                        |
//   900  | altcallback kernel eval| not defined




#define BADZEROTOL 1e-12
#define BADVARTOL 1e-3

class MercerKernel;

std::ostream &operator<<(std::ostream &output, const MercerKernel &src );
std::istream &operator>>(std::istream &input,        MercerKernel &dest);
int operator==(const MercerKernel &leftop, const MercerKernel &rightop);






#define DEFAULT_XPROD_SIZE 2

class vecInfoBase;


inline void qswap(vecInfoBase &a, vecInfoBase &b);
inline void qswap(const vecInfoBase *&a, const vecInfoBase *&b);
inline void qswap(vecInfoBase *&a, vecInfoBase *&b);

inline vecInfoBase &setident (vecInfoBase &a);
inline vecInfoBase &setzero  (vecInfoBase &a);
inline vecInfoBase &setposate(vecInfoBase &a);
inline vecInfoBase &setnegate(vecInfoBase &a);
inline vecInfoBase &setconj  (vecInfoBase &a);
inline vecInfoBase &setrand  (vecInfoBase &a);

inline vecInfoBase *&setident (vecInfoBase *&a);
inline vecInfoBase *&setzero  (vecInfoBase *&a);
inline vecInfoBase *&setposate(vecInfoBase *&a);
inline vecInfoBase *&setnegate(vecInfoBase *&a);
inline vecInfoBase *&setconj  (vecInfoBase *&a);
inline vecInfoBase *&setrand  (vecInfoBase *&a);

inline const vecInfoBase *&setident (const vecInfoBase *&a);
inline const vecInfoBase *&setzero  (const vecInfoBase *&a);
inline const vecInfoBase *&setposate(const vecInfoBase *&a);
inline const vecInfoBase *&setnegate(const vecInfoBase *&a);
inline const vecInfoBase *&setconj  (const vecInfoBase *&a);
inline const vecInfoBase *&setrand  (const vecInfoBase *&a);

OVERLAYMAKEFNVECTOR(vecInfoBase)
OVERLAYMAKEFNVECTOR(Vector<vecInfoBase>)
OVERLAYMAKEFNVECTOR(SparseVector<vecInfoBase>)

class vecInfoBase
{
public:

    svm_explicit vecInfoBase()
    {
        // Initialise values such that any normalisation has no effect

        xhalfmprod.resize(DEFAULT_XPROD_SIZE/2);

        xiseqn = 0;

//        xmean   = 0.0;
//        xmedian = 0.0;
//        xsqmean = 0.0;
//        xvari   = 1.0;
//        xstdev  = 1.0;
//        xmax    = 1.0;
//        xmin    = 0.0;
//        xrange  = 1.0;
//        xmaxabs = 1.0;

        xhalfinda = &xhalfindb;
        xhalfindb = &xhalfmprod;

        xusize = 1;

        hasbeenset = 0;

        return;
    }

    vecInfoBase(const vecInfoBase &src)
    {
        *this = src;

        xhalfinda = &xhalfindb;
        xhalfindb = &xhalfmprod;

        hasbeenset = src.hasbeenset;

        return;
    }

    vecInfoBase &operator=(const vecInfoBase &src)
    {
        xhalfmprod = src.xhalfmprod;

        xiseqn = src.xiseqn;

//        xmean   = src.xmean;
//        xmedian = src.xmedian;
//        xsqmean = src.xsqmean;
//        xvari   = src.xvari;
//        xstdev  = src.xstdev;
//        xmax    = src.xmax;
//        xmin    = src.xmin;
//        xrange  = src.xrange;
//        xmaxabs = src.xmaxabs;

        xusize = src.xusize;

        hasbeenset = src.hasbeenset;

        return *this;
    }

    Vector<gentype> xhalfmprod;

    int xiseqn;

//    gentype xmean;
//    gentype xmedian;
//    gentype xsqmean;
//    gentype xvari;
//    gentype xstdev;
//    gentype xmax;
//    gentype xmin;
//    gentype xrange;
//    gentype xmaxabs;

    int xusize;

    Vector<gentype> **xhalfinda;
    Vector<gentype> *xhalfindb;

    int hasbeenset;
};

inline vecInfoBase &setident (vecInfoBase &a) { throw("no"); return a; }
inline vecInfoBase &setposate(vecInfoBase &a) { return a; }
inline vecInfoBase &setnegate(vecInfoBase &a) { throw("bleh"); return a; }
inline vecInfoBase &setconj  (vecInfoBase &a) { throw("blit"); return a; }
inline vecInfoBase &setrand  (vecInfoBase &a) { throw("OK, rand"); return a; }

inline vecInfoBase *&setident (vecInfoBase *&a) { throw("no"); return a; }
inline vecInfoBase *&setzero  (vecInfoBase *&a) { return a = NULL; }
inline vecInfoBase *&setposate(vecInfoBase *&a) { return a; }
inline vecInfoBase *&setnegate(vecInfoBase *&a) { throw("bleh"); return a; }
inline vecInfoBase *&setconj  (vecInfoBase *&a) { throw("blit"); return a; }
inline vecInfoBase *&setrand  (vecInfoBase *&a) { throw("OK, rand"); return a; }

inline const vecInfoBase *&setident (const vecInfoBase *&a) { throw("no"); return a; }
inline const vecInfoBase *&setzero  (const vecInfoBase *&a) { return a = NULL; }
inline const vecInfoBase *&setposate(const vecInfoBase *&a) { return a; }
inline const vecInfoBase *&setnegate(const vecInfoBase *&a) { throw("bleh"); return a; }
inline const vecInfoBase *&setconj  (const vecInfoBase *&a) { throw("blit"); return a; }
inline const vecInfoBase *&setrand  (const vecInfoBase *&a) { throw("OK, rand"); return a; }

inline vecInfoBase &setzero(vecInfoBase &a)
{
    a.xhalfmprod.resize(DEFAULT_XPROD_SIZE/2);

    a.xiseqn = 0;

//    a.xmean   = 0.0;
//    a.xmedian = 0.0;
//    a.xsqmean = 0.0;
//    a.xvari   = 1.0;
//    a.xstdev  = 1.0;
//    a.xmax    = 1.0;
//    a.xmin    = 0.0;
//    a.xrange  = 1.0;
//    a.xmaxabs = 1.0;

    a.xhalfinda = &(a.xhalfindb);
    a.xhalfindb = &(a.xhalfmprod);

    a.xusize = 1;

    a.hasbeenset = 0;

    return a;
}

inline void qswap(const vecInfoBase *&a, const vecInfoBase *&b)
{
    const vecInfoBase *c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(vecInfoBase *&a, vecInfoBase *&b)
{
    vecInfoBase *c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(vecInfoBase &a, vecInfoBase &b)
{
    qswap(a.xhalfmprod,b.xhalfmprod);
    qswap(a.xiseqn    ,b.xiseqn    );

//    qswap(a.xmean  ,b.xmean  );
//    qswap(a.xmedian,b.xmedian);
//    qswap(a.xsqmean,b.xsqmean);
//    qswap(a.xvari  ,b.xvari  );
//    qswap(a.xstdev ,b.xstdev );
//    qswap(a.xmax   ,b.xmax   );
//    qswap(a.xmin   ,b.xmin   );
//    qswap(a.xrange ,b.xrange );
//    qswap(a.xmaxabs,b.xmaxabs);
//    qswap(a.xusize ,b.xusize );

    qswap(a.hasbeenset,b.hasbeenset);

    return;
}

inline void qswap(SparseVector<vecInfoBase> *&a, SparseVector<vecInfoBase> *&b);
inline void qswap(SparseVector<vecInfoBase> *&a, SparseVector<vecInfoBase> *&b)
{
    SparseVector<vecInfoBase> *c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}

class vecInfo;

inline void qswap(vecInfo &a, vecInfo &b);
inline void qswap(const vecInfo *&a, const vecInfo *&b);
inline void qswap(vecInfo *&a, vecInfo *&b);

OVERLAYMAKEFNVECTOR(vecInfo)
OVERLAYMAKEFNVECTOR(Vector<vecInfo>)
OVERLAYMAKEFNVECTOR(SparseVector<vecInfo>)

inline vecInfo &getvecscratch(void);

class vecInfo
{
public:
    svm_explicit vecInfo()
    {
        content.resize(2);

        int z = 0;

        MEMNEW(content("&",z),SparseVector<vecInfoBase>);
        MEMNEW(content("&",1),SparseVector<vecInfoBase>);

        (*(content("&",z)))("&",z);
        (*(content("&",1)))("&",z);

        isloc = 1;

        minind = 0;
        majind = 0;

        usize_overwrite = 0;

        return;
    }

    svm_explicit vecInfo(const vecInfoBase &src)
    {
        content.resize(2);

        int z = 0;

        MEMNEW(content("&",z),SparseVector<vecInfoBase>);
        MEMNEW(content("&",1),SparseVector<vecInfoBase>);

        (*(content("&",z)))("&",z);
        (*(content("&",1)))("&",z);

        isloc = 1;

        minind = 0;
        majind = 0;

        (*(content("&",z)))("&",z) = src;

        usize_overwrite = 0;

        return;
    }

    vecInfo(const vecInfo &src)
    {
        content.resize(2);

        isloc = 0;

        minind = 0;
        majind = 0;

        usize_overwrite = 0;

        *this = src;

        return;
    }

    vecInfo &operator()(int _majind = -1, int _minind = -1, int xusize_overwrite = 0) const
    {
        vecInfo &res = getvecscratch();

        int z = 0;

        if ( res.isloc )
        {
            MEMDEL((res.content)("&",z));
            MEMDEL((res.content)("&",1));

            (res.isloc) = 0;
        }

        (res.content).resize(2);

        (res.majind) = ( _majind == -1 ) ? majind : _majind;
        (res.minind) = ( _minind == -1 ) ? minind : _minind;

        (res.content)("&",z) = content(z);
        (res.content)("&",1) = content(1);

        (res.usize_overwrite) = xusize_overwrite ? xusize_overwrite : usize_overwrite;

        return res;
    }

    vecInfo &operator=(const vecInfo &src)
    {
        int z = 0;

        if ( isloc && src.isloc )
        {
            MEMDEL(content("&",z));
            MEMDEL(content("&",1));

            MEMNEW(content("&",z),SparseVector<vecInfoBase>);
            MEMNEW(content("&",1),SparseVector<vecInfoBase>);

            (*(content("&",z))) = (*((src.content(z))));
            (*(content("&",1))) = (*((src.content(1))));
        }

        else if ( !isloc && src.isloc )
        {
            MEMNEW(content("&",z),SparseVector<vecInfoBase>);
            MEMNEW(content("&",1),SparseVector<vecInfoBase>);

            (*(content("&",z))) = (*((src.content(z))));
            (*(content("&",1))) = (*((src.content(1))));
        }

        if ( isloc && !(src.isloc) )
        {
            MEMDEL(content("&",z));
            MEMDEL(content("&",1));

            content("&",z) = src.content(z);
            content("&",1) = src.content(1);
        }

        else
        {
            content("&",z) = src.content(z);
            content("&",1) = src.content(1);
        }

        isloc = src.isloc;

        minind = src.minind;
        majind = src.majind;

        usize_overwrite = src.usize_overwrite;

        return *this;
    }

    ~vecInfo()
    {
        if ( isloc )
        {
            int z = 0;

            MEMDEL(content("&",z));
            MEMDEL(content("&",1));
        }

        return;
    }

    const Vector<gentype> &xhalfmprod(void) const { return relbase().xhalfmprod; }

    const int &xiseqn(void) const { return relbase().xiseqn; }

//    const gentype &xmean  (void) const { return relbase().xmean;   }
//    const gentype &xmedian(void) const { return relbase().xmedian; }
//    const gentype &xsqmean(void) const { return relbase().xsqmean; }
//    const gentype &xvari  (void) const { return relbase().xvari;   }
//    const gentype &xstdev (void) const { return relbase().xstdev;  }
//    const gentype &xmax   (void) const { return relbase().xmax;    }
//    const gentype &xmin   (void) const { return relbase().xmin;    }
//    const gentype &xrange (void) const { return relbase().xrange;  }
//    const gentype &xmaxabs(void) const { return relbase().xmaxabs; }

    const int &xusize(void) const { return usize_overwrite ? usize_overwrite : relbase().xusize; }

    Vector<gentype> **xhalfinda(void) const { return relbase().xhalfinda; }
    Vector<gentype>  *xhalfindb(void) const { return relbase().xhalfindb; }

    const int &hasbeenset(void) const { return relbase().hasbeenset; }

    Vector<gentype> &xhalfmprod(void) { return relbase().xhalfmprod; }

    int &xiseqn(void) { return relbase().xiseqn; }

//    gentype &xmean  (void) { return relbase().xmean;   }
//    gentype &xmedian(void) { return relbase().xmedian; }
//    gentype &xsqmean(void) { return relbase().xsqmean; }
//    gentype &xvari  (void) { return relbase().xvari;   }
//    gentype &xstdev (void) { return relbase().xstdev;  }
//    gentype &xmax   (void) { return relbase().xmax;    }
//    gentype &xmin   (void) { return relbase().xmin;    }
//    gentype &xrange (void) { return relbase().xrange;  }
//    gentype &xmaxabs(void) { return relbase().xmaxabs; }

    int &xusize(void) { return usize_overwrite ? usize_overwrite : relbase().xusize; }

    int &hasbeenset(void) { return relbase().hasbeenset; }

    const vecInfoBase &relbase(void) const { return (*(content(    majind)))(    minind); }
          vecInfoBase &relbase(void)       { return (*(content("&",majind)))("&",minind); }

//private: - ok whatever, but don't use them

    Vector<SparseVector<vecInfoBase> *> content;

    int isloc;

    int minind;
    int majind;

    int usize_overwrite;
};

inline vecInfo &setident (vecInfo &a) { throw("no"); return a; }
inline vecInfo &setzero  (vecInfo &a) { vecInfo b; return a = b; }
inline vecInfo &setposate(vecInfo &a) { return a; }
inline vecInfo &setnegate(vecInfo &a) { throw("bleh"); return a; }
inline vecInfo &setconj  (vecInfo &a) { throw("blit"); return a; }
inline vecInfo &setrand  (vecInfo &a) { throw("OK, rand"); return a; }

inline vecInfo *&setident (vecInfo *&a) { throw("no"); return a; }
inline vecInfo *&setzero  (vecInfo *&a) { return a = NULL; }
inline vecInfo *&setposate(vecInfo *&a) { return a; }
inline vecInfo *&setnegate(vecInfo *&a) { throw("bleh"); return a; }
inline vecInfo *&setconj  (vecInfo *&a) { throw("blit"); return a; }
inline vecInfo *&setrand  (vecInfo *&a) { throw("OK, rand"); return a; }

inline const vecInfo *&setident (const vecInfo *&a) { throw("no"); return a; }
inline const vecInfo *&setzero  (const vecInfo *&a) { return a = NULL; }
inline const vecInfo *&setposate(const vecInfo *&a) { return a; }
inline const vecInfo *&setnegate(const vecInfo *&a) { throw("bleh"); return a; }
inline const vecInfo *&setconj  (const vecInfo *&a) { throw("blit"); return a; }
inline const vecInfo *&setrand  (const vecInfo *&a) { throw("OK, rand"); return a; }




inline vecInfo &getvecscratch(void)
{
    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    static Vector<vecInfo> scratch(VECINFOSCRATCHSIZE);
    svmvolatile static int indind = 0;

    vecInfo &res = scratch("&",indind);

    indind++;

    if ( indind >= VECINFOSCRATCHSIZE )
    {
        indind = 0;
    }

//Original code:
//indind = (indind+1)%VECINFOSCRATCHSIZE
//Initially I was confused why this didn't work but the 
//replacement code works just fine.  Then I realised that
//it was because the macro was 4*DEFAULT_NUM_TUPLES, so
//this expanded to (indind+1)%4*DEFAULT_NUM_TUPLES, which
//by order of operation is equivalent to 
//((indind+1)%4)*DEFAULT_NUM_TUPLES... which is not
//what I want at all.  I could easily fix this but the
//new code works just fine and it was such a pain to track
//down I honestly don't want to mess with it.

    svm_mutex_unlock(eyelock);

    return res;
}


inline void qswap(const vecInfo *&a, const vecInfo *&b)
{
    const vecInfo *c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(vecInfo *&a, vecInfo *&b)
{
    vecInfo *c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(vecInfo &a, vecInfo &b)
{
    qswap(a.content        ,b.content        );
    qswap(a.isloc          ,b.isloc          );
    qswap(a.minind         ,b.minind         );
    qswap(a.majind         ,b.majind         );
    qswap(a.usize_overwrite,b.usize_overwrite);

    return;
}




// Kernel re-entry prototype: you can inherit from this and then overwrite
// K2xfer with some other function to evaluate the kernel.  Then set the
// pointers to force callback that over-rides ktype

class kernPrecursor;

inline kernPrecursor *&setzero  (kernPrecursor *&a);
inline kernPrecursor *&setident (kernPrecursor *&a);
inline kernPrecursor *&setposate(kernPrecursor *&a);
inline kernPrecursor *&setnegate(kernPrecursor *&a);
inline kernPrecursor *&setconj  (kernPrecursor *&a);
inline kernPrecursor *&setrand  (kernPrecursor *&a);

OVERLAYMAKEFNVECTOR(kernPrecursor)
OVERLAYMAKEFNVECTOR(Vector<kernPrecursor>)
OVERLAYMAKEFNVECTOR(SparseVector<kernPrecursor>)


inline void qswap(kernPrecursor *&a, kernPrecursor *&b);

class kernPrecursor
{
public:
    svm_explicit kernPrecursor()
    {
        svm_mutex_lock(kerneyelock); 

        if ( fullmllist == NULL )
        {
            SparseVector<kernPrecursor *> *locfullmllist = NULL;

            MEMNEW(locfullmllist,SparseVector<kernPrecursor *>);

            NiceAssert(locfullmllist);

            fullmllist = locfullmllist;
        }

        xmlid = xmlidcnt(); 

        (*const_cast<SparseVector<kernPrecursor *>*>(fullmllist))("&",xmlid) = this; 

        svm_mutex_unlock(kerneyelock); 

        return;
    }

    virtual ~kernPrecursor()
    {
        svm_mutex_lock(kerneyelock); 

        (*const_cast<SparseVector<kernPrecursor *>*>(fullmllist)).remove(xmlid); 

        if ( !((*const_cast<SparseVector<kernPrecursor *>*>(fullmllist)).indsize()) )
        {
            MEMDEL(const_cast<SparseVector<kernPrecursor *>*>(fullmllist));

            fullmllist = NULL;
        }

        svm_mutex_unlock(kerneyelock); 
    }

    kernPrecursor &operator=(const kernPrecursor &src)
    {
        assign(src);

        return *this;
    }

    virtual void assign(const kernPrecursor &src, int onlySemiCopy = 0)
    {
        (void) src;
        (void) onlySemiCopy;

        return;
    }

    virtual void semicopy(const kernPrecursor &src)
    {
        (void) src;

        return;
    }

    virtual void qswapinternal(kernPrecursor &b)
    {
        int nv;

        nv = xmlid; xmlid = b.xmlid; b.xmlid = nv;

        return;
    }

    virtual int isKVarianceNZ(void) const
    {
        return 0;
    }

    //
    // - resmode = 0: (default) the result is a number (or matrix or whatever)
    //   This is almost (but not quite) like the definition below, but with
    //   additional points at end.
    //
    // NB: - d2K support removed, placeholder resmode left
    //     - dkdr likewise removed, placeholder left
    //     - modes 16 and 32 also calculate result
    //
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | resmode | x,y    | integer consts | real consts | calculate | calculate | calculate | calculate | calculate  |
    // | resmode | subbed |     subbed     |    subbed   |   dk/dr   | dk/dxnorm | dk/dxyprod| d2k/dzdz  | K variance |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 0 deflt |   y    |        y       |      y      |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 1       |        |        y       |      y      |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 2       |   y    |                |      y      |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 3       |        |                |      y      |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 4       |   y    |        y       |             |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 5       |        |        y       |             |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 6       |   y    |                |             |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 7       |        |                |             |           |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 8       |   y    |        y       |      y      |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 9       |        |        y       |      y      |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 10      |   y    |                |      y      |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 11      |        |                |      y      |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 12      |   y    |        y       |             |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 13      |        |        y       |             |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 14      |   y    |                |             |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 15      |        |                |             |     y     |           |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 16      |   y    |        y       |      y      |           |     y     |           |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 32      |   y    |        y       |      y      |           |           |     y     |           |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 64      |   y    |        y       |      y      |           |           |           |     y     |            |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+
    // | 128     |   y    |        y       |      y      |           |           |           |           |     y      |
    // +---------+--------+----------------+-------------+-----------+-----------+-----------+-----------+------------+

//NB: templates cannot be made virtual, so need both versions

    virtual void K0xfer(gentype &res, int &minmaxind, int typeis,
                       int xdim, int densetype, int resmode, int mlid) const
    {
        (void) minmaxind;
        (void) typeis;
        (void) xdim;
        (void) densetype;
        (void) resmode;
        (void) mlid;

        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //throw("K0xfer not defined here for m = 4");

        res = 0.0;

        return;
    }

    virtual void K0xfer(double &res, int &minmaxind, int typeis,
                       int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;

        K0xfer(tempres,minmaxind,typeis,xdim,densetype,resmode,mlid);

        res = (double) tempres;

        return;
    }

    virtual void K1xfer(gentype &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, 
                       const vecInfo &xainfo, 
                       int ia, 
                       int xdim, int densetype, int resmode, int mlid) const
    {
        (void) minmaxind;
        (void) typeis;
        (void) xa;
        (void) xainfo;
        (void) ia;
        (void) xdim;
        (void) densetype;
        (void) resmode;
        (void) mlid;

        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //throw("K1xfer not defined here for m = 4");

        res = 0.0;

        return;
    }

    virtual void K1xfer(double &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, 
                       const vecInfo &xainfo, 
                       int ia, 
                       int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;

        K1xfer(tempres,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

        res = (double) tempres;

        return;
    }

    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
    {
        (void) minmaxind;
        (void) typeis;
        (void) xa;
        (void) xb;
        (void) xainfo;
        (void) xbinfo;
        (void) ia;
        (void) ib;
        (void) xdim;
        (void) densetype;
        (void) resmode;
        (void) mlid;

        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //throw("K2xfer not defined here for m = 2");

        res = 0.0;

        dxyprod = 0.0;
        ddiffis = 0.0;

        return;
    }

    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;
        gentype tempdxyprod;
        gentype tempddiffis;

        K2xfer(tempdxyprod,tempddiffis,tempres,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

        res = (double) tempres;

        dxyprod = tempdxyprod;
        ddiffis = tempddiffis;

        return;
    }

    virtual void K3xfer(gentype &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                       int ia, int ib, int ic, 
                       int xdim, int densetype, int resmode, int mlid) const
    {
        (void) minmaxind;
        (void) typeis;
        (void) xa;
        (void) xb;
        (void) xc;
        (void) xainfo;
        (void) xbinfo;
        (void) xcinfo;
        (void) ia;
        (void) ib;
        (void) ic;
        (void) xdim;
        (void) densetype;
        (void) resmode;
        (void) mlid;

        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //throw("K3xfer not defined here for m = 4");

        res = 0.0;

        return;
    }

    virtual void K3xfer(double &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                       int ia, int ib, int ic, 
                       int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;

        K3xfer(tempres,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

        res = (double) tempres;

        return;
    }

    virtual void K4xfer(gentype &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                       int ia, int ib, int ic, int id,
                       int xdim, int densetype, int resmode, int mlid) const
    {
        (void) minmaxind;
        (void) typeis;
        (void) xa;
        (void) xb;
        (void) xc;
        (void) xd;
        (void) xainfo;
        (void) xbinfo;
        (void) xcinfo;
        (void) xdinfo;
        (void) ia;
        (void) ib;
        (void) ic;
        (void) id;
        (void) xdim;
        (void) densetype;
        (void) resmode;
        (void) mlid;

        // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
        //throw("K4xfer not defined here for m = 4");

        res = 0.0;

        return;
    }

    virtual void K4xfer(double &res, int &minmaxind, int typeis,
                       const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                       const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                       int ia, int ib, int ic, int id,
                       int xdim, int densetype, int resmode, int mlid) const
    {
        gentype tempres;

        K4xfer(tempres,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

        res = (double) tempres;

        return;
    }

    virtual void Kmxfer(gentype &res, int &minmaxind, int typeis,
                       Vector<const SparseVector<gentype> *> &x,
                       Vector<const vecInfo *> &xinfo,
                       Vector<int> &i,
                       int xdim, int m, int densetype, int resmode, int mlid) const
    {
        if ( m == 0 )
        {
            K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid);
        }

        else if ( m == 1 )
        {
            K1xfer(res,minmaxind,typeis,*x(zeroint()),*xinfo(zeroint()),i(zeroint()),xdim,densetype,resmode,mlid);
        }

        else if ( m == 2 )
        {
            gentype dummy;

            K2xfer(dummy,dummy,res,minmaxind,typeis,*x(zeroint()),*x(1),*xinfo(zeroint()),*xinfo(1),i(zeroint()),i(1),xdim,densetype,resmode,mlid);
        }

        else if ( m == 3 )
        {
            K3xfer(res,minmaxind,typeis,*x(zeroint()),*x(1),*x(2),*xinfo(zeroint()),*xinfo(1),*xinfo(2),i(zeroint()),i(1),i(2),xdim,densetype,resmode,mlid);
        }

        else if ( m == 4 )
        {
            K4xfer(res,minmaxind,typeis,*x(zeroint()),*x(1),*x(2),*x(3),*xinfo(zeroint()),*xinfo(1),*xinfo(2),*xinfo(3),i(zeroint()),i(1),i(2),i(3),xdim,densetype,resmode,mlid);
        }

        else
        {
            res = 0.0;

            // Design decision: just return 0 for now for simplicity (allowing -kt 8xx -ktx 0 to work) and let ml_base take care of rest
            //throw("Kmxfer not defined here for m > 4");
        }

        return;
    }

    virtual void Kmxfer(double &res, int &minmaxind, int typeis,
                       Vector<const SparseVector<gentype> *> &x,
                       Vector<const vecInfo *> &xinfo,
                       Vector<int> &i,
                       int xdim, int m, int densetype, int resmode, int mlid) const
    {
        if ( m == 0 )
        {
            K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid);
        }

        else if ( m == 1 )
        {
            K1xfer(res,minmaxind,typeis,*x(zeroint()),*xinfo(zeroint()),i(zeroint()),xdim,densetype,resmode,mlid);
        }

        else if ( m == 2 )
        {
            double dummy;

            K2xfer(dummy,dummy,res,minmaxind,typeis,*x(zeroint()),*x(1),*xinfo(zeroint()),*xinfo(1),i(zeroint()),i(1),xdim,densetype,resmode,mlid);
        }

        else if ( m == 3 )
        {
            K3xfer(res,minmaxind,typeis,*x(zeroint()),*x(1),*x(2),*xinfo(zeroint()),*xinfo(1),*xinfo(2),i(zeroint()),i(1),i(2),xdim,densetype,resmode,mlid);
        }

        else if ( m == 4 )
        {
            K4xfer(res,minmaxind,typeis,*x(zeroint()),*x(1),*x(2),*x(3),*xinfo(zeroint()),*xinfo(1),*xinfo(2),*xinfo(3),i(zeroint()),i(1),i(2),i(3),xdim,densetype,resmode,mlid);
        }

        else
        {
            gentype tempres(res);

            Kmxfer(tempres,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

            res = (double) tempres;
        }

        return;
    }





    // Kernel transfer switching stuff.  All MLs are registered so you
    // can switch between them (kernel transfer).
    //
    // MLid(): unique ID for this this ML.
    // setMLid(): MLid is default > INT_MAX/2 - use this to set to more sensible
    //            value.  Return 0 on success, nz otherwise.
    // getaltML(): get reference to ML with given ID.  Return 0 on success, 1 if NULL.

    virtual int MLid(void) const
    {
        return xmlid;
    }

    virtual int setMLid(int nv)
    {
        int res = 0;

        svm_mutex_lock(kerneyelock);

        if ( nv < 0 )
        {
            res = 1;
        }

        else if ( (*const_cast<SparseVector<kernPrecursor *>*>(fullmllist)).isindpresent(nv) )
        {
            res = 2;
        }

        else
        {
            (*const_cast<SparseVector<kernPrecursor *>*>(fullmllist)).zero(xmlid);

            xmlid = nv;

            (*const_cast<SparseVector<kernPrecursor *>*>(fullmllist))("&",xmlid) = this;
        }

        svm_mutex_unlock(kerneyelock);

        return res;
    }

    virtual int getaltML(kernPrecursor *&res, int altMLid) const
    {
        svm_mutex_lock(kerneyelock);

        int ires = 1;
        res = NULL;

        if ( (*const_cast<SparseVector<kernPrecursor *>*>(fullmllist)).isindpresent(altMLid) )
        {
            ires = 0;
            res = (*const_cast<SparseVector<kernPrecursor *>*>(fullmllist))("&",altMLid);
        }

        svm_mutex_unlock(kerneyelock);

        return ires;
    }

    int xmlid;

    virtual int type(void) const { return -2; }

private:

    int &xmlidcnt(void)
    {
        static int loccnt = INT_MAX/2; 

        return ++loccnt;
    }

    svmvolatile static SparseVector<kernPrecursor *>* fullmllist;
    svmvolatile static svm_mutex kerneyelock; 
};

inline kernPrecursor *&setident (kernPrecursor *&a) { throw("no"); return a; }
inline kernPrecursor *&setposate(kernPrecursor *&a) { return a; }
inline kernPrecursor *&setnegate(kernPrecursor *&a) { throw("bleh"); return a; }
inline kernPrecursor *&setconj  (kernPrecursor *&a) { throw("blit"); return a; }
inline kernPrecursor *&setrand  (kernPrecursor *&a) { throw("OK, rand"); return a; }

inline kernPrecursor *&setzero(kernPrecursor *&a)
{
    return a = NULL;
}

inline void qswap(kernPrecursor *&a, kernPrecursor *&b)
{
    kernPrecursor *c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}






//
// Kernel information structure: this stores information about what info
// the kernel function uses to evaluate.  Adding these gives the flags if
// the 

class kernInfo;

OVERLAYMAKEFNVECTOR(kernInfo)
OVERLAYMAKEFNVECTOR(Vector<kernInfo>)
OVERLAYMAKEFNVECTOR(SparseVector<kernInfo>)

inline void qswap(kernInfo *&a, kernInfo *&b);

class kernInfo
{
public:

    kernInfo()
    {
        usesDiff    = 0;
        usesInner   = 0;
        usesNorm    = 0;
        usesVector  = 0;
        usesMinDiff = 0;
        usesMaxDiff = 0;

        return;
    }

    kernInfo(const kernInfo &src)
    {
        usesDiff    = src.usesDiff;
        usesInner   = src.usesInner;
        usesNorm    = src.usesNorm;
        usesVector  = src.usesVector;
        usesMinDiff = src.usesMinDiff;
        usesMaxDiff = src.usesMaxDiff;

        return;
    }

    kernInfo &operator=(const kernInfo &src)
    {
        usesDiff    = src.usesDiff;
        usesInner   = src.usesInner;
        usesNorm    = src.usesNorm;
        usesVector  = src.usesVector;
        usesMinDiff = src.usesMinDiff;
        usesMaxDiff = src.usesMaxDiff;

        return *this;
    }

    int numflagsset(void) const
    {
        return usesDiff+usesInner+usesNorm+usesVector+usesMinDiff+usesMaxDiff;
    }

    kernInfo &zero(void)
    {
        usesDiff    = 0;
        usesInner   = 0;
        usesNorm    = 0;
        usesVector  = 0;
        usesMinDiff = 0;
        usesMaxDiff = 0;

        return *this;
    }

    unsigned int usesDiff    : 1; // set if kernel uses ||x-y||^2 explicitly
    unsigned int usesInner   : 1; // set if kernel uses x'y explicitly
    unsigned int usesNorm    : 1; // set if kernel uses ||x||^2 and ||y||^2 explicitly
    unsigned int usesVector  : 1; // set if kernel uses x and y vectors explicitly
    unsigned int usesMinDiff : 1; // set if kernel uses min(x-y) explicitly
    unsigned int usesMaxDiff : 1; // set if kernel uses max(x-y) explicitly
};

inline void qswap(kernInfo *&a, kernInfo *&b)
{
    kernInfo *c = NULL;

    c = a;
    a = b;
    b = c;

    return;
}

//
// +=: this is defined so that summing a vector of kernInfo works as OR
// ==: equivalence operator
// <<: output stream
// >>: input strea,
// setzero: sets all flags 0
// qswap:   standard quickswap operation
//

kernInfo &operator+=(kernInfo &a, const kernInfo &b);
int operator==(const kernInfo &a, const kernInfo &b);

std::ostream &operator<<(std::ostream &output, const kernInfo &src);
std::istream &operator>>(std::istream &input, kernInfo &dest);

inline kernInfo &setzero  (kernInfo &a);
inline kernInfo &setident (kernInfo &a);
inline kernInfo &setposate(kernInfo &a);
inline kernInfo &setnegate(kernInfo &a);
inline kernInfo &setconj  (kernInfo &a);
inline kernInfo &setrand  (kernInfo &a);

inline kernInfo &postProInnerProd(kernInfo &a) { return a; }
inline void qswap(kernInfo &a, kernInfo &b);

inline kernInfo &setident (kernInfo &a) { throw("no"); return a; }
inline kernInfo &setposate(kernInfo &a) { return a; }
inline kernInfo &setnegate(kernInfo &a) { throw("bleh"); return a; }
inline kernInfo &setconj  (kernInfo &a) { throw("blit"); return a; }
inline kernInfo &setrand  (kernInfo &a) { throw("OK, rand"); return a; }

inline kernInfo *&setident (kernInfo *&a) { throw("no"); return a; }
inline kernInfo *&setzero  (kernInfo *&a) { return a = NULL; }
inline kernInfo *&setposate(kernInfo *&a) { return a; }
inline kernInfo *&setnegate(kernInfo *&a) { throw("bleh"); return a; }
inline kernInfo *&setconj  (kernInfo *&a) { throw("blit"); return a; }
inline kernInfo *&setrand  (kernInfo *&a) { throw("OK, rand"); return a; }

inline const kernInfo *&setident (const kernInfo *&a) { throw("no"); return a; }
inline const kernInfo *&setzero  (const kernInfo *&a) { return a = NULL; }
inline const kernInfo *&setposate(const kernInfo *&a) { return a; }
inline const kernInfo *&setnegate(const kernInfo *&a) { throw("bleh"); return a; }
inline const kernInfo *&setconj  (const kernInfo *&a) { throw("blit"); return a; }
inline const kernInfo *&setrand  (const kernInfo *&a) { throw("OK, rand"); return a; }

inline kernInfo &setzero(kernInfo &a)
{
    return a.zero();
}

inline void qswap(kernInfo &a, kernInfo &b)
{
    kernInfo c(a);

    a = b;
    b = c;

    return;
}






std::ostream &operator<<(std::ostream &output, const vecInfoBase &src);
std::istream &operator>>(std::istream &input, vecInfoBase &dest);

std::ostream &operator<<(std::ostream &output, const vecInfo &src);
std::istream &operator>>(std::istream &input, vecInfo &dest);

// Swap function

inline void qswap(MercerKernel &a, MercerKernel &b);

OVERLAYMAKEFNVECTOR(MercerKernel)
OVERLAYMAKEFNVECTOR(Vector<MercerKernel>)
OVERLAYMAKEFNVECTOR(SparseVector<MercerKernel>)

class MercerKernel : public kernPrecursor
{
    friend std::ostream &operator<<(std::ostream &output, const MercerKernel &src );
    friend std::istream &operator>>(std::istream &input,        MercerKernel &dest);
    friend int operator==(const MercerKernel &leftop, const MercerKernel &rightop);
    friend void qswap(MercerKernel &a, MercerKernel &b);

public:

    virtual int getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const;

    // Constructors and assignment operators

    svm_explicit MercerKernel();
                 MercerKernel(const MercerKernel &src);

    ~MercerKernel();

    MercerKernel &operator=(const MercerKernel &src);

    // Information:
    //
    // isIndex:              x'z is indexed (x'z -> sum_{i in S} x_i.y_i)
    // isShifted:            vectors shifted (x -> (x-sh))
    // isScaled:             vectors scaled (x -> diag(sc).x)
    // isShiftedScaled:      vectors shifted and scaled (x -> diag(sc).(x-sh))
    // isLeftPlain:          no normalisation applied to x in K(x,y)
    // isRightPlain:         no normalisation applied to y in K(x,y)
    // isLeftRightPlain:     no normalisation applied to x and y in K(x,y)
    // isLeftNormal:         normalisation may be applied to x in K(x,y)
    // isRightNormal:        normalisation may be applied to y in K(x,y)
    // isLeftRightNormal:    normalisation may be applied to x and y in K(x,y)
    // isPartNormal:         normalisation may be applied to x or y in K(x,y)
    // isProd:               K(x,y) = prod_i K(x_i,y_i)
    // isNormalised:         set if K_q is scaled/shifted (x -> diag(sc).(x-sh))
    // isAltDiff:            0:   ||x-y||_2^2    -> ||x||_m^m + ||y||_m^m + ... - m.<<x,y,...>>_m
    //                       1:   ||x-y||_2^2    -> ||x||_2^2 + ||y||_2^2 + ... - 2.<<x,y,...>>_m (default)
    //                       2:   2*(||x-y||_2^2 -> ||x||_2^2 + ||y||_2^2 + ... - (1/m).(sum_{ij} <xi,xj>))
    //                            (the RBF has additional scaling as per paper - see Kbase)
    //                       103: K(...) -> 1/2^{m-1} \sum_{s = [ +-1 +-1 ... ] \in R^m : |i:si=+1| + |i:si=-1| \in 4Z_+} K(||sum_i s_i x_i ||_2^2)
    //                       104: K(...) -> 1/m!      \sum_{s = [ +-1 +-1 ... ] \in R^m : |i:si=+1| = |i:si=-1|         } K(||sum_i s_i x_i ||_2^2)
    //                       203: like 103, but kernel expansion occurs over first kernel in chain only
    //                       204: like 104, but kernel expansion occurs over first kernel in chain only
    //                       300: true moment-kernel expension to 2-kernels
    // needsmProd:           0: m-kernel calculation does not require <<x,y,...>>_m
    //                       1: m-kernel calculation requires <<x,y,...>>_m
    // wantsXYprod:          0: providing an xy matrix will not result in speedup
    //                       1: providing an xy matrix will result in speedup
    // suggestXYcache:       0: suggest to user not pass xy matrix for 2-kernel cache.
    //                       1: suggest to user that passing xy matrix will help even in 2-kernel cache.
    //                       (this is purely advisory and can be ignored)
    // isIPdiffered:         0: inner-products are probably unchanged
    //                       1: inner-products have changed
    //
    // See below for information.  Note that SVM kernels are always
    // LeftRightNormal, and NN kernels are always LeftNormal / RightPlain
    //
    // size: number of kernels in complete kernel
    //
    // cIndexes: index vector S if used
    // cShift:   shift used for normalisation
    // cScale:   scale used for normalisation
    //
    // cWeight(q):      weight w_q for K_q
    // cType(q):        type of K_q
    // isChained(q):    see below:
    //
    // Normally the total kernel function is:
    //
    // K = K_0 + K_1 + K_2 + K_3 + K_4 ...
    //
    // chaining involves taking the output of one kernel and feeding to the
    // input of the next kernel.  So if, for example, isChained(1) &&
    // isChained(2):
    //
    // K = K_0 + K_3(K_2(K_1)) + K_4 + ...
    //
    // Note that chaining will only work for kernels that do not explicitly
    // use x,y (rather than x'x, x'y and y'y).  Kernels for which
    // !isKkitchensink(q) are fine.
    //
    // cRealConstants(q): real constants for K_q
    // cIntConstants(q):  integer constants for K_q
    // cRealOverwrite(q): real constant overwrites for K_q (see below)
    // cIntOverwrite(q):  integer constant overwrites for K_q (see below)
    //
    // isSplit(q): set if kernel is "split" (separated into multiple kernels for different parts of x1,x2,...) here
    // magTerm(q): rather than K(x,y), evaluate K(x,x).K(y,y)
    //
    // Constant overwrites let you take the value for real (integer)
    // constants from the input vectors x and y.  For example, if
    // cRealOverwrite(q) = ( 0:2 1:10 ) then:
    //
    // realConstant(0) -> x(2)*y(2)    (x(2) is rightPlain, y(2) if leftPlain)
    // realConstant(1) -> x(10)*y(10)  (x(2) is rightPlain, y(2) if leftPlain)

    int isFullNorm       (void) const { return isfullnorm;                }
    int isProd           (void) const { return isprod;                    }
    int isIndex          (void) const { return isind;                     }
    int isShifted        (void) const { return isshift & 1;               }
    int isScaled         (void) const { return isshift & 2;               }
    int isShiftedScaled  (void) const { return isshift == 3;              }
    int isLeftPlain      (void) const { return leftplain;                 }
    int isRightPlain     (void) const { return rightplain;                }
    int isLeftRightPlain (void) const { return leftplain && rightplain;   }
    int isLeftNormal     (void) const { return !leftplain;                }
    int isRightNormal    (void) const { return !rightplain;               }
    int isLeftRightNormal(void) const { return !leftplain && !rightplain; }
    int isPartNormal     (void) const { return !leftplain || !rightplain; }
    int isAltDiff        (void) const { return isdiffalt;                 }
    int needsmProd       (void) const { return needsInner(-1,4);          }
    int wantsXYprod      (void) const { return needsMatDiff();            }
    int suggestXYcache   (void) const { return xsuggestXYcache;           }
    int isIPdiffered     (void) const { return xisIPdiffered;             }

    int size(void)        const { return isnorm.size(); }
    int getSymmetry(void) const; // 1 for symmetric, -1 for anti, 0 for none

    const       Vector<int>     &cIndexes(void) const { return dIndexes; }
    const SparseVector<gentype> &cShift  (void) const { return dShift;   }
    const SparseVector<gentype> &cScale  (void) const { return dScale;   }

    const gentype &cWeight(int q = 0) const { return dRealConstants(q)(0); }
    int            cType  (int q = 0) const { return dtype(q);             }

    int isNormalised(int q = 0) const { return isnorm(q);    }
    int isChained   (int q = 0) const { return ischain(q);   }
    int isSplit     (int q = 0) const { return issplit(q);   }
    int isMulSplit  (int q = 0) const { return mulsplit(q);  }
    int isMagTerm   (int q = 0) const { return ismagterm(q); }

    int numSplits(void)    const { return xnumSplits;    }
    int numMulSplits(void) const { return xnumMulSplits; }

          int              numSamples        (void) const { return xnumsamples; }
    const Vector<gentype> &sampleDistribution(void) const { return xsampdist;   }
    const Vector<int>     &sampleIndices     (void) const { return xindsub;     }

    const kernPrecursor *getAltCall  (int q = 0) const { kernPrecursor *res = NULL; int ires = getaltML(res,altcallback(q)); NiceAssert( !ires ); (void) ires; return res; }
          kernPrecursor *getAltCallnc(int q = 0)       { kernPrecursor *res = NULL; int ires = getaltML(res,altcallback(q)); NiceAssert( !ires ); (void) ires; return res; }

    const Vector<gentype>   &cRealConstants(int q = 0) const { return dRealConstants(q)(1,1,dRealConstants(q).size()-1,const_cast<retVector<gentype> &>(cRealConstantsTmp)); }
    const Vector<int>       &cIntConstants (int q = 0) const { return dIntConstants(q);                                                                                      }
    const SparseVector<int> &cRealOverwrite(int q = 0) const { return dRealOverwrite(q);                                                                                     }
    const SparseVector<int> &cIntOverwrite (int q = 0) const { return dIntOverwrite(q);                                                                                      }

    const gentype &getRealConstZero(int q = 0) const { return cRealConstants(q)(zeroint()); }
    const int     &getIntConstZero (int q = 0) const { return cIntConstants(q)(zeroint());  }

    // isKVarianceNZ: does K have variance (ie. inheritted from some GP or similar that uses averaging)

    int isKVarianceNZ(void) const
    {
        int res = 0;

        if ( size() )
        {
            int i;

            for ( i = 0 ; i < size() ; i++ )
            {
                if ( ( cType(i) >= 800 ) && ( cType(i) <= 999 ) )
                {
                    if ( (*(getAltCall(i))).isKVarianceNZ() )
                    {
                        res = 1;
                        break;
                    }
                }
            }
        }

        return res;
    }

    // More detailed kernel information
    //
    // Note that this does not take norming into account.

    const kernInfo &kinf(int q) const { return kernflags(q); }
    const kernInfo  kinf(void)  const { return sum(kernflags); }

    // Modifiers:

    MercerKernel &add   (int q);
    MercerKernel &remove(int q);
    MercerKernel &resize(int nsize);

    MercerKernel &setFullNorm       (void) {                                                                                      isfullnorm       = 1;                                  return *this; }
    MercerKernel &setNoFullNorm     (void) {                                                                                      isfullnorm       = 0;                                  return *this; }
    MercerKernel &setProd           (void) {                    xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isprod           = 1;                                  return *this; }
    MercerKernel &setnonProd        (void) {                    xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isprod           = 0;                                  return *this; }
    MercerKernel &setLeftPlain      (void) { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; leftplain        = 1;                  fixShiftProd(); return *this; }
    MercerKernel &setRightPlain     (void) { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; rightplain       = 1;                  fixShiftProd(); return *this; }
    MercerKernel &setLeftRightPlain (void) { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; leftplain        = 1;  rightplain = 1; fixShiftProd(); return *this; }
    MercerKernel &setLeftNormal     (void) { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; leftplain        = 0;                  fixShiftProd(); return *this; }
    MercerKernel &setRightNormal    (void) { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; rightplain       = 0;                  fixShiftProd(); return *this; }
    MercerKernel &setLeftRightNormal(void) { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; leftplain        = 0;  rightplain = 0; fixShiftProd(); return *this; }

    MercerKernel &setAltDiff       (int nv) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isdiffalt        = nv; return *this; }
    MercerKernel &setsuggestXYcache(int nv) {                                                                   xsuggestXYcache  = nv; return *this; }
    MercerKernel &setIPdiffered    (int nv) {                                                                   xisIPdiffered    = nv; return *this; }

    MercerKernel &setnumSamples        (int                    nv) { xnumsamples = nv; return *this; }
    MercerKernel &setSampleDistribution(const Vector<gentype> &nv) { xsampdist   = nv; return *this; }
    MercerKernel &setSampleIndices     (const Vector<int>     &nv) { xindsub     = nv; return *this; }

    MercerKernel &setIndexes(const Vector<int> &ndIndexes) { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isind = 1; dIndexes = ndIndexes; fixShiftProd(); return *this; }
    MercerKernel &setUnIndex(void)                         { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isind = 0;                       fixShiftProd(); return *this; }

    MercerKernel &setShift(const SparseVector<gentype> &ndShift) { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isshift |= 1; dShift = ndShift; dShift.makealtcontent(); fixShiftProd(); return *this; }
    MercerKernel &setScale(const SparseVector<gentype> &ndScale) { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isshift |= 2; dScale = ndScale; dScale.makealtcontent(); fixShiftProd(); return *this; }
    MercerKernel &setUnShiftedScaled(void)                       { xisIPdiffered = 1; xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isshift  = 0;                                            fixShiftProd(); return *this; }

    MercerKernel &setChained   (int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ischain("&",q)   = 1;                                     return *this; }
    MercerKernel &setNormalised(int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isnorm("&",q)    = 1;                                     return *this; }
    MercerKernel &setSplit     (int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; issplit("&",q)   = 1; xnumSplits    = calcnumSplits();    return *this; }
    MercerKernel &setMulSplit  (int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; mulsplit("&",q)  = 1; xnumMulSplits = calcnumMulSplits(); return *this; }
    MercerKernel &setMagTerm   (int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ismagterm("&",q) = 1;                                     return *this; }

    MercerKernel &setUnChained   (int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ischain("&",q)   = 0;                                     return *this; }
    MercerKernel &setUnNormalised(int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; isnorm("&",q)    = 0;                                     return *this; }
    MercerKernel &setUnSplit     (int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; issplit("&",q)   = 0; xnumSplits    = calcnumSplits();    return *this; }
    MercerKernel &setUnMulSplit  (int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; mulsplit("&",q)  = 0; xnumMulSplits = calcnumMulSplits(); return *this; }
    MercerKernel &setUnMagTerm   (int q = 0) { xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; ismagterm("&",q) = 0;                                     return *this; }

    MercerKernel &setWeight (const gentype &nw, int q = 0) { dRealConstants("&",q)("&",0) = nw; return *this; }
    MercerKernel &setType   (int ndtype,        int q = 0);
    MercerKernel &setAltCall(int newMLid,       int q = 0) { kill900channel(q); xisfast = -1; xneedsInner = -1; xneedsDiff = -1; xneedsNorm = -1; altcallback("&",q) = newMLid; return *this; }

    MercerKernel &setRealConstants(const Vector<gentype>   &ndRealConstants, int q = 0);
    MercerKernel &setIntConstants (const Vector<int>       &ndIntConstants,  int q = 0);
    MercerKernel &setRealOverwrite(const SparseVector<int> &ndRealOverwrite, int q = 0);
    MercerKernel &setIntOverwrite (const SparseVector<int> &ndIntOverwrite,  int q = 0);

    MercerKernel &setRealConstZero(double nv, int q = 0);
    MercerKernel &setIntConstZero (int nv,    int q = 0);

    // Element retrieval
    //
    // Sets res = x(i).direcref(j) (shifted/scaled if shifting/scaling is turned on)
    // and returns reference to this.

    gentype &xelm(gentype &res, const SparseVector<gentype> &x, int i, int j) const;
    int xindsize(const SparseVector<gentype> &x, int i) const;
//    const SparseVector<gentype> &getx(const SparseVector<gentype> &x, int i) const { (void) i; return x; }

    // Kernel-space distance
    //
    // ||x-y|| = K(x,x)+K(y,y)-2K(x,y)
    //
    // but may be accelerated for some cases (kernels 300-399)

    inline double distK(const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int assumreal = 0) const;
    inline void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int assumreal = 0) const;

    // Evaluate kernel K(x,y).
    //
    // Optional arguments: xnorm = x'x
    //                     ynorm = y'y
    //
    // If normalisation on then returns K(x,y)/sqrt(K(x,x)*K(y,y))
    //
    // Matrix arguments:
    //
    // - rows of left-hand matrix argument are x vectors
    // - columns of right-hand matrix argument are x vectors
    // - allrow forms assume arguments are rows in both left and right hand
    // - allcol forms assume arguments are columns in both left and right hand
    //
    // Biased forms:
    //
    // - bias is added to the inner product
    // - BiasedR implies bias vector is replaced by 1*b'
    // - BiasedL implies bias vector is replaced by b*1'
    //
    // Vector-less forms:
    //
    // - Only work for kernels that do not require explicit use of vectors
    //   x and y.  All pre-normalisation etc is assumed taken care of.
    // - biased: it is assumed that the bias HAS NOT BEEN ADDED to either
    //   xyprod or yxprod.
    // - real vectorless forms are very fast for isSimpleNNKernel types.
    //   These are the only K functions optimised for this type of kernel.
    //
    // Indexing:
    //
    // If xconsist is set then it is assumed that x and y share the same
    // indexes, which allows certain optimisations to occur (namely inner
    // products can be more quickly calculated).
    //
    // pxyprod: - if xyprod and diffis known then make this an array where
    //            pxyprod[0] points to xyprod and pxyprod[1] points to diffis
    //          - if xyprod is known only then " but pxyprod[1] = NULL
    //          - if diffis is known only then " but pxyprod[0] = NULL
    //          - otherwise set NULL
    //          - for 8xx kernel pxyprod is the result
    //          - for simple chained 8xx kernels pxyprod is the result of the first part
    //
    //
    // Return as equation option (Keqn):
    //
    // - resmode = 0: (default) the result is a number (or matrix or whatever)
    //   K variance: if K is inherited then it can have inherent variance.
    //
    // +---------+--------+----------------+-------------+-----------+------------+
    // | resmode | x,y    | integer consts | real consts | calculate | calculate  |
    // | resmode | subbed |     subbed     |    subbed   |   dk/dr   | K variance |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 0 deflt |   y    |        y       |      y      |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 1       |        |        y       |      y      |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 2       |   y    |                |      y      |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 3       |        |                |      y      |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 4       |   y    |        y       |             |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 5       |        |        y       |             |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 6       |   y    |                |             |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 7       |        |                |             |           |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 8       |   y    |        y       |      y      |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 9       |        |        y       |      y      |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 10      |   y    |                |      y      |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 11      |        |                |      y      |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 12      |   y    |        y       |             |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 13      |        |        y       |             |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+ 
    // | 14      |   y    |                |             |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 15      |        |                |             |     y     |            |
    // +---------+--------+----------------+-------------+-----------+------------+
    // | 128     |   y    |        y       |      y      |           |     y      |
    // +---------+--------+----------------+-------------+-----------+------------+
    //
    // - in equation: var(0,0) = x'x (if !(resmode|1))
    //                var(0,1) = y'y (if !(resmode|1))
    //                var(0,2) = x'y (if !(resmode|1))
    //                var(2,i) = ij  (if !(resmode|2))
    //                var(1,j) = rj  (if !(resmode|4))
    // - this can also do real constant gradients when resmode|8.  In this
    //   case the result is a vector of the required dimension, the elements
    //   of which may or may not be equations depending on the settings of
    //   the three LSB of resmode (see above table).

    gentype &Keqn(gentype &res, int resmode = 1) const
    {
        const static SparseVector<gentype> x;
        const static SparseVector<gentype> y;

        const static vecInfo xinfo;
        const static vecInfo yinfo;

        K2(res,x,y,xinfo,yinfo,defaultgentype(),NULL,DEFAULT_VECT_INDEX,DEFAULT_VECT_INDEX,0,0,resmode,0,NULL,NULL,NULL,0);

        return res;
    }

    // Kernels for different norms.
    //
    // NB: - odd-order kernels implemented for fast kernels only.
    //     - The vectorial form can speed up k2xfer forms
    //     - for 2-norm forms:
    //
    // xy matrix stores either inner products [ <x,x> <x,y> ; <y,x> <y,y> ] or their transferred
    // equivalents [ K2xfer(x,x) K2xfer(x,y) ; K2xfer(y,x) K2xfer(y,y) ].
    //
    // xconsist:  set if we can assume the indices of x,y, etc and scale/shift are all the same
    // assumreal: set 1 if we assume x is real-valued to speed things up (call before doing kernel eval)

    inline gentype &K0(gentype &res, const gentype &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal = 0) const;
    inline double  &K0(double  &res, const double  &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal = 0) const;

    inline gentype &K1(gentype &res, const SparseVector<gentype> &x, const vecInfo &xinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = NULL, int assumreal = 0) const;
    inline double  &K1(double  &res, const SparseVector<gentype> &x, const vecInfo &xinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = NULL, int assumreal = 0) const;

    inline gentype &K2(gentype &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int assumreal = 0) const;
    inline double  &K2(double  &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int assumreal = 0) const;

    inline gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int k = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, const double *xy20 = NULL, const double *xy21 = NULL, const double *xy22 = NULL, int assumreal = 0) const;
    inline double  &K3(double  &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int k = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, const double *xy20 = NULL, const double *xy21 = NULL, const double *xy22 = NULL, int assumreal = 0) const;

    inline gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int k = DEFAULT_VECT_INDEX, int l = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, const double *xy20 = NULL, const double *xy21 = NULL, const double *xy22 = NULL, const double *xy30 = NULL, const double *xy31 = NULL, const double *xy32 = NULL, const double *xy33 = NULL, int assumreal = 0) const;
    inline double  &K4(double  &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int k = DEFAULT_VECT_INDEX, int l = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, const double *xy20 = NULL, const double *xy21 = NULL, const double *xy22 = NULL, const double *xy30 = NULL, const double *xy31 = NULL, const double *xy32 = NULL, const double *xy33 = NULL, int assumreal = 0) const;

    inline gentype &Km(int m, gentype &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const gentype &bias, Vector<int> &i, const gentype **pxyprod = NULL, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const Matrix<double> *xy = NULL, int assumreal = 0) const;
    inline double  &Km(int m, double  &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const double  &bias, Vector<int> &i, const gentype **pxyprod = NULL, int xdim = 0, int xconsist = 0, int resmode = 0, int mlid = 0, const Matrix<double> *xy = NULL, int assumreal = 0) const;

    // Feature maps
    //
    // phim: returns the image of x in feature space.  This may be finite dimensional if possible 
    //       (and allowfinite = 1), but otherwise infinite dimensional.  Does not include bias term.
    // phidim: return -1 if feature map infinite dimensional, >= 0 otherwise

    Vector<gentype> &phi1(Vector<gentype> &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(1,res,x,i,allowfinite,xdim,xconsist,assumreal); }
    Vector<double>  &phi1(Vector<double>  &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(1,res,x,i,allowfinite,xdim,xconsist,assumreal); }

    Vector<gentype> &phi2(Vector<gentype> &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(2,res,x,i,allowfinite,xdim,xconsist,assumreal); }
    Vector<double>  &phi2(Vector<double>  &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(2,res,x,i,allowfinite,xdim,xconsist,assumreal); }

    Vector<gentype> &phi3(Vector<gentype> &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(3,res,x,i,allowfinite,xdim,xconsist,assumreal); }
    Vector<double>  &phi3(Vector<double>  &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(3,res,x,i,allowfinite,xdim,xconsist,assumreal); }

    Vector<gentype> &phi4(Vector<gentype> &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(4,res,x,i,allowfinite,xdim,xconsist,assumreal); }
    Vector<double>  &phi4(Vector<double>  &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const { return phim(4,res,x,i,allowfinite,xdim,xconsist,assumreal); }

    Vector<gentype> &phim(int m, Vector<gentype> &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const;
    Vector<double>  &phim(int m, Vector<double>  &res, const SparseVector<gentype> &x, int i = DEFAULT_VECT_INDEX, int allowfinite = 1, int xdim = 0, int xconsist = 0, int assumreal = 0) const;

    int phidim(int allowfinite = 1, int xdim = 0) const;

    // Inner-product calculation forms
    //
    // These just calculate the inner product, not the kernel.  If *simple* kernel transfer
    // is enabled then the inner product is actually the result of evaluating the transfered
    // kernel.  This operation is not well defined for non-simple kernels.

    inline double &K0ip(double &res, const double &bias, const gentype **pxyprod, int xdim, int xconsist, int mlid, int assumreal) const;
    inline double &K1ip(double &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, const double &bias, const gentype **pxyprod = NULL, int ia = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    inline double &K2ip(double &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const double &bias, const gentype **pxyprod = NULL, int ia = DEFAULT_VECT_INDEX, int ib = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    inline double &K3ip(double &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const double &bias, const gentype **pxyprod = NULL, int ia = DEFAULT_VECT_INDEX, int ib = DEFAULT_VECT_INDEX, int ic = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    inline double &K4ip(double &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const double &bias, const gentype **pxyprod = NULL, int ia = DEFAULT_VECT_INDEX, int ib = DEFAULT_VECT_INDEX, int ic = DEFAULT_VECT_INDEX, int id = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    inline double &Kmip(int m, double &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, const double &bias, const gentype **pxyprod = NULL, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;

    // 2-norm derivatives
    //
    // dK2delx: calculates derivative w.r.t. x vector.  Result is xscaleres.x + yscaleres.y.
    // d2K2delxdelx: calculates 2nd derivative d/dx d/dx K.  Result is xxscaleres.x.x' + yyscaleres.y.y' + xyscaleres.x.y' + yxscaleres.y.x' + constres.I
    // d2K2delxdely: calculates 2nd derivative d/dx d/dy K.  Result is xxscaleres.x.x' + yyscaleres.y.y' + xyscaleres.x.y' + yxscaleres.y.x' + constres.I
    // dnK2del: nth derivative (currently only for RBF kernel).  Result is an array:
    //
    // dnK/dx_q0.dx_q1... sum_i sc_i kronProd_{j=0,1,...} [ x{n_ij}   if n_ij = 0,1
    //                                                    [ kd{n_ij}  if n_ij < 0
    //
    // where: x{0} = x
    //        x{1} = y
    //        kd{a} ... kd{a} = kronecker-delta (vectorised identity matrix) on indices
    //
    // If minmaxind >= 0 then derivative is only with respect to element minmaxind
    // of vectors x,y (so result is xscaleres.x(minmaxind) + yscaleres.y(minmaxind)).
    //
    // NB: this is actually the derivative wrt dScale.*(x-dShift) if shifting and/or
    //     scaling is present, so factor this in when calculating results.  That is
    //     d/dx_i => dScale_i d/dx_i etc

    inline void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int assumreal = 0) const;
    inline void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int assumreal = 0) const;

    inline void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDerive = 0, int assumreal = 0) const;
    inline void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDerive = 0, int assumreal = 0) const;

    inline void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDerive = 0, int assumreal = 0) const;
    inline void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDerive = 0, int assumreal = 0) const;

    inline void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDerive = 0, int assumreal = 0) const;
    inline void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDerive = 0, int assumreal = 0) const;

    // 2-norm derivatives (alternative form - deprecated)
    //
    // dK: assuming kernel can be written K(<x,y>,||x||^2,||y||^2), with symmetry in x,y, this
    //     returns the derivatives with respect to the first two arguments.
    //     If K is a simple transfer kernel then this derivative is with respect to the the
    //     arguments in the form K(K2xfer(x,y),K2xfer(x,x),K2xfer(y,y)) (the derivative 
    //     dK/dK2xfer(x,y), xK/dK2xfer(x,x)).  This behaviour can be changed by setting the 
    //     arguments deepDeriv to 1, which will recurse down to <x,y>,||x||^2,||y||^2 where
    //     possible.  Behaviour for non-simple transfer kernels is ill-defined (unless
    //     deepDeriv is set to 1).

    inline void dK(gentype &xygrad, gentype &xnormgrad, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDeriv = 0, int assumreal = 0) const;
    inline void dK(double  &xygrad, double  &xnormgrad, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDeriv = 0, int assumreal = 0) const;

    // Note: dk(x,y)/dynorm = dk(y,x)/dxnorm etc, standard assumptions necessary

    inline void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDeriv = 0, int assumreal = 0) const;
    inline void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double  &bias, const gentype **pxyprod = NULL, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, const double *xy00 = NULL, const double *xy10 = NULL, const double *xy11 = NULL, int deepDeriv = 0, int assumreal = 0) const;

    // "Reversing" functions.
    // 
    // For speed of operation it is sometimes helpful to retrieve either the
    // inner product or distance from an evaluated kernel.  These functions
    // let you do that
    // 
    // isReversible: test if kernel is reversible.  Output is:
    //     0: kernel cannot be reversed
    //     1: kernel can be reversed to produce <x,y>+bias
    //     2: kernel can be reversed to produce ||x-y||^2
    // 
    // reverseK: reverse kernel as described by isReversible
    // 
    // The result so produced can be fed back in via the pxyprod argument
    // (appropriately set) to speed up calculation of results.  Use case
    // could be quickly changing kernel parameters with minimal recalculation.
    //
    // As a general rule these only work with isSimpleFastKernel or 
    // isSimpleKernelChain, and then in limited cases.  For the chain case
    // the result is the relevant (processed) output of the first layer.

    inline int isReversible(void) const;
    inline gentype &reverseK(gentype &res, const gentype &Kval) const;
    inline double  &reverseK(double &res,  const double  &Kval) const;

    // Evaluate kernel gradient dK/dx(x,y) and dK/dy(x,y)
    //
    // FIXME: at present this assumes everything is real-valued.  The more
    // general case is a little more difficult.
    //
    // Product kernels are not dealt with at present
    //
    // The returned value is in terms of x and y scales.  The gradient so
    // represented is of the form (x gradient case):
    //
    // dK/dx = xscaleres.x + yscaleres.y
    //
    // densedKdx: for product kernels only this calculates:
    //            d/dx0 d/dx1 ... K(x,y) = \prod_j dK(xj,yj)/dxj
    //            *This will disable callback for inner product calculation*
    // denseintK: reverse of densedKdx
    //
    // Design decision: for the kernel defined on max(x_k-y_k) the dense
    // derivative is simply the derivative on this axis.  This enables us
    // to estimate variance on the pareto frontier.
    //
    // At present this makes the following assumptions:
    //
    // - the caller is aware of any indexing tricks
    // - kernels are either inner product or norm difference kernels
    //
    // Biased gradients: these actually return gradients for the vectors
    //
    // ( x )  and  ( y    )
    // ( 1 )       ( bias )
    //
    // (so xscaleres refers to the scale for the augmented vector, and like-
    // wise yscaleres).  This makes surprisingly little difference if you
    // want the gradients for x and y as the scale factors are the same
    // (dxaug/dx = diag(I,0)).  To calculate the bias gradient, note that:
    //
    // dK/dbias = dK/dxaug dxaug/dbias + dK/dyaug dyaug/dbias
    //          = dK/dyaug dyaug/dbias
    //          = dyxscaleres + bias.dyyscaleres
    //
    // minmaxind: -1  if gradient is for whole x/y
    //            >=0 if gradient is for just one element of min/max(x-y)
    //
    // For Km gradients: xyscaleres refers to scaling factor on x(0).^{m-1} (for dx) or x(1).^{m-1} (for dy) (.^ is the elementwise power)
    //                   zscaleres refers to scaling factor on x(i0)*x(i1)*...x(i{m-2}) (elementwise product of all x except x(0) (for dx) or x(1) (for dy))
    // 
    // Currently second order gradients are not implemented for Km kernels.  If
    // you want to implement them bear in mind that you will need xxscaleres for
    // weighting (m-1)-order terms (and yyscaleres etc), plus constres, but also
    // weight terms for (m-2) order terms, so things will get a bit tricky.
    //
    // Only a very limited subset of second order derivatives are defined!

    // Dense derivatives and integrals

    inline void densedKdx(double &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double &bias, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;
    inline void denseintK(double &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double &bias, int i = DEFAULT_VECT_INDEX, int j = DEFAULT_VECT_INDEX, int xdim = 0, int xconsist = 0, int mlid = 0, int assumreal = 0) const;

    // Get vector information, taking into account indexing.
    //
    // scratch: may or may not be used for something or other, saves on allocs and statics
    //
    // xmag: 2-norm, if known

    vecInfo     &getvecInfo(vecInfo     &res, const SparseVector<gentype> &x, const gentype *xmag = NULL, int xconsist = 0, int assumreal = 0) const;
    vecInfoBase &getvecInfo(vecInfoBase &res, const SparseVector<gentype> &x, const gentype *xmag = NULL, int xconsist = 0, int assumreal = 0) const;

    inline const gentype &getmnorm(const vecInfo &xinfo, const SparseVector<gentype> &x, int m, int xconsist = 0, int assumreal = 0) const;
    inline       gentype &getmnorm(      vecInfo &xinfo, const SparseVector<gentype> &x, int m, int xconsist = 0, int assumreal = 0) const;

    // Kernel scaling

    MercerKernel &operator*=(double sf)
    {
        int q;

        for ( q = 0 ; q < size() ; q++ )
        {
            if ( !isChained(q) || isSplit(q) || isMulSplit(q) )
            {
                dRealConstants("&",q)("&",0) *= sf;
            }

            if ( isSplit(q) || isMulSplit(q) )
            {
                break;
            }
        }

        return *this;
    }

private:

    int calcnumSplits(int indstart = 0, int indend = -1) const
    {
        retVector<int> tmp;

        if ( indend == -1 )
        {
            indend = size()-1;
        }

        return ( indend <= 0 ) ? 0 : sum(issplit(indstart,1,indend-1,tmp));
    }

    int calcnumMulSplits(int indstart = 0, int indend = -1) const
    {
        retVector<int> tmp;

        if ( indend == -1 )
        {
            indend = size()-1;
        }

        return ( indend <= 0 ) ? 0 : sum(mulsplit(indstart,1,indend-1,tmp));
    }

    // Terms used:
    //
    // - normalised:         Kn(x,y) = K(x,y)/sqrt(|K(x,x)|.|K(y,y)|)
    // - shifted and scaled: Ks(x,y) = K((x+shift).*scale,(y+shift).*scale)
    //
    // dtype: kernel type vector.
    // isprod:     0 = normal, 1 = K(x,y) = prod_i K(x_i,y_i)
    // isnorm:     0 = normal, 1 = normalisation on.
    // isdiffalt:  see previous
    // ischain:    0 = normal, 1 = this kernel is then chained into next kernel.
    // issplit:    0 = normal, 1 = this kernel (for this part of x) stops here, next kernel (for this part of x) starts.
    // mulsplit:   0 = normal, 1 = this kernel (for all x) stops here, next kernel (for all x) starts.
    // ismagtern   0 = normal, 1 = use K(x,x).K(y,y) rather than K(x,y).
    // isshift:    0 = normal, 1 = use shifting only, 2 = use scaling only, 3 = use shifting and scaling
    // isind:      0 = normal, 1 = use indexed products
    // isfullnorm: 0 = normal, 1 = normalise at outermost
    // leftplain:  0 = normal, 1 = don't shift-scale left-hand argument in K
    // rightplain: 0 = normal, 1 = don't shift-scale right-hand argument in K
    // weight: weight factor kernel is multiplied by
    //         (now stored as index 0 of real constants)
    // dIntConstants: integer constants in kernel
    // dRealConstants: real constants in kernel
    // dIntOverwrite: selects which variables will be overwritten by which x(i)*y(i)
    // dRealOverwrite: selects which variables will be overwritten by which x(i)*y(i)
    // dIndexes: indices used in index products
    // dShift: shift factor
    // dScale: scale factor
    // dShiftProd: ||dShift.*dScale||_2^2

    int isind;
    int isshift;
    int leftplain;
    int rightplain;
    int isprod;
    int isdiffalt;
    int xproddepth;
    int enchurn; // set if kernel reversal is enabled.
    int xsuggestXYcache;
    int xisIPdiffered;
    int isfullnorm;
    int xnumSplits;
    int xnumMulSplits;

    Vector<int> dtype;
    Vector<int> isnorm;
    Vector<int> ischain;
    Vector<int> issplit;
    Vector<int> mulsplit;
    Vector<int> ismagterm;
    Vector<int> dIndexes;
    Vector<kernInfo> kernflags;
    Vector<Vector<gentype> > dRealConstants;
    Vector<Vector<int> > dIntConstants;
    Vector<SparseVector<int> > dRealOverwrite;
    Vector<SparseVector<int> > dIntOverwrite;
    Vector<int> altcallback;

    SparseVector<gentype> dShift;
    SparseVector<gentype> dScale;
    gentype dShiftProd;
    gentype dShiftProdNoConj;
    gentype dShiftProdRevConj;

    retVector<gentype> cRealConstantsTmp;

    // Feature no longer used, assume enchurn == 0
    //
    // churnInner:   set 1 if we want to attempt to retrieve and reuse 
    //               inner products <x,y>+b and distances ||x-y||^2 when
    //               changing the kernel (see prepareKernel in ml_base).  
    //               This does not guarantee that retrieval will occur, but
    //               only that if it is possible and implemented then it will
    //               be attempted when feasible.  Only really speeds things up
    //               when you're using kernel inheritance.
    //

    int churnInner(void) const { return enchurn; }
    MercerKernel &setChurnInner(int nv) { enchurn = nv; return *this; }

    // Distribution kernel information:
    //
    // xnumSamples: number of samples to estimate the kernel
    // xindsub:     indices of variables in distributions that are substituted
    // xsampdist:   sample distribution for these variables
    //
    // That is, we are estimating:
    //
    // E_a[K(x(a),y(a))]
    //
    // where x and y are (or contain) *distributions*, so
    //
    // x \sim dist(a)
    //
    // and a itself if drawn from sampdist, and has indices xindsub.

    int xnumsamples;
    Vector<int> xindsub;
    Vector<gentype> xsampdist;

    // Call tree to calculate kernel.
    //
    // - public version calculates xyprod etc and calls first version
    //
    // - yyyK version does preprocessing on [ xa ~ xb ... ] forms
    // - next levels goes through kernel structure and calls second version
    //   to evaluate individual kernels in structure (indexed by q)
    // - this calls unnorm form and applies normalisation if needed
    // - unnorm form calculates diffis (||x-y||^2) if needed and then
    //   calls Kbase
    // - Kbase does the actual work.
    // - resmode = 0: standard evaluation
    //   resmode = 1: return equation, including constants
    //   resmode = 2: return equation, integers substituted out
    //   resmode = 3: return equation, integers and reals subbed out
    //     var(0,0) = x'x
    //     var(0,1) = y'y
    //     var(0,2) = x'y
    //     var(1,j) = rj (real constants) (if resmode == 1,2)
    //     var(2,i) = ij (integer constants) (if resmode == 1)
    //
    // K4 and Km are similar but work with 4 and m norms before finally
    // converging onto the same Kbase for final evaluation
    //
    // densetype = 0: normal operation
    //             1: calculate dense derivative d/dx0 d/dx1 ... K(x,y)
    //             2: calculate dense integral int_x0 ind_x1 ... K(x,y)
    //
    // iset: controls how distributions in the data are treated.
    //     0 = standard behaviour, distributions are treated as such, result is average of samples
    //     1 = distributions represent draws from infinite sets, result is largest (most similar) evaluation with draws from given distribution(s)
    //
    // The actual dense derivative/integral operations work by pairing
    // kernels.  That is, if K is the kernel then the dense derivative will
    // find the kernel corresponding to the derivative and then call that
    // instead.  Only works if required pair is defined.
    //
    // FIXME: will also only work for simple kernels

    template <class T> T &yyyK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyyK1(T &res, const SparseVector<gentype> &x, const vecInfo &xinfo, const T &bias, const gentype **pxyprod, int i, int xdim, int xconsist, int resmode, int mlid, const double *xy, int assumreal, int justcalcip) const;
    template <class T> T &yyyK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const;
    template <class T> T &yyyK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;
    template <class T> T &yyyK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const;
    template <class T> T &yyyKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, int assumreal, int justcalcip) const;

    template <class T> T &yyyaK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyyaK1(T &res, const SparseVector<gentype> &x, const vecInfo &xinfo, const T &bias, const gentype **pxyprod, int i, int xdim, int xconsist, int resmode, int mlid, const double *xy, int assumreal, int justcalcip) const;
    template <class T> T &yyyaK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const;
    template <class T> T &yyyaK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const;
    template <class T> T &yyyaK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const;
    template <class T> T &yyyaKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, int assumreal, int justcalcip) const;

    template <class T> T &yyybK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyybK1(T &res, const SparseVector<gentype> &x, const vecInfo &xinfo, const T &bias, const gentype **pxyprod, int i, int xdim, int xconsist, int resmode, int mlid, const double *xy, int iset, int assumreal, int justcalcip) const;
    template <class T> T &yyybK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int iset, int jset, int assumreal, int justcalcip) const;
    template <class T> T &yyybK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iaset, int ibset, int icset, int assumreal, int justcalcip) const;
    template <class T> T &yyybK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int assumreal, int justcalcip) const;
    template <class T> T &yyybKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *iset, int assumreal, int justcalcip) const;
    template <class T> T &yyybKmb(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, Vector<int> &imask, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *iset, int assumreal, int justcalcip) const;

    template <class T> T &yyycK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyycK1(T &res, const SparseVector<gentype> &x, const vecInfo &xinfo, const T &bias, const gentype **pxyprod, int i, int xdim, int xconsist, int resmode, int mlid, const double *xy, int iset, int assumreal, int justcalcip) const;
    template <class T> T &yyycK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int iset, int jset, int assumreal, int justcalcip) const;
    template <class T> T &yyycK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iaset, int ibset, int icset, int assumreal, int justcalcip) const;
    template <class T> T &yyycK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int assumreal, int justcalcip) const;
    template <class T> T &yyycKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *iset, int assumreal, int justcalcip) const;

    template <class T> T &yyyKK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, int assumreal, int justcalcip) const;
    template <class T> T &yyyKK1(T &res, const SparseVector<gentype> &x, const vecInfo &xinfo, const T &bias, const gentype **pxyprod, int i, int xdim, int xconsist, int resmode, int mlid, const double *xy, int iset, int assumreal, int justcalcip) const;
    template <class T> T &yyyKK2(T &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int iset, int jset, int assumreal, int justcalcip) const;
    template <class T> T &yyyKK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iaset, int ibset, int icset, int assumreal, int justcalcip) const;
    template <class T> T &yyyKK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int i, int j, int k, int l, int xdim, int xconsist, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int assumreal, int justcalcip) const;
    template <class T> T &yyyKKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *iset, int assumreal, int justcalcip) const;

    template <class T> T &xKKK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip) const;
    template <class T> T &xKKK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int iaset) const;
    template <class T> T &xKKK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int justcalcip, int iset, int jset) const;
    template <class T> T &xKKK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int justcalcip, int iaset, int ibset, int icset) const;
    template <class T> T &xKKK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int justcalcip, int iaset, int ibset, int icset, int idset) const;
    template <class T> T &xKKKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, const Matrix<double> *xy, int justcalcip, const Vector<int> *iset) const;

    template <class T> T &xKK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip, int indstart, int indend, int ns) const;
    template <class T> T &xKK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int iaset, int indstart, int indend, int ns) const;
    template <class T> T &xKK2(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy00, const double * xy10, const double *xy11, int justcalcip, int iset, int jset, int indstart, int indend, int ns) const;
    template <class T> T &xKK3(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int justcalcip, int iaset, int ibset, int icset, int indstart, int indend, int ns) const;
    template <class T> T &xKK4(T &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int indstart, int indend, int ns) const;
    template <class T> T &xKKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, const Matrix<double> *xy, int justcalcip, const Vector<int> *iset, int indstart, int indend, int ns) const;

    template <class T> T &KK0(T &res, T &logres, int &logresvalid, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip, int indstart, int indend, int skipbias = 0) const;
    template <class T> T &KK1(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const vecInfo &xainfo, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int indstart, int indend, int iaset, int skipbias = 0, int skipxa = 0) const;
    template <class T> T &KK2(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int justcalcip, int indstart, int indend, int iset, int jset, int skipbias = 0, int skipxa = 0, int skipxb = 0) const;
    template <class T> T &KK3(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const Vector<int> *s, int justcalcip, int indstart, int indend, int iaset, int ibset, int icset, int skipbias = 0, int skipxa = 0, int skipxb = 0, int skipxc = 0) const;
    template <class T> T &KK4(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const Vector<int> *s, int justcalcip, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int indstart, int indend, int iaset, int ibset, int icset, int idset, int skipbias = 0, int skipxa = 0, int skipxb = 0, int skipxc = 0, int skipxd = 0) const;
    template <class T> T &KK6(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const SparseVector<gentype> &xe, const SparseVector<gentype> &xf, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const vecInfo &xeinfo, const vecInfo &xfinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int ie, int jf, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const Matrix<double> *xy, const Vector<int> *s, int justcalcip, int indstart, int indend, int iaset, int ibset, int icset, int idset, int ieset, int ifset) const;
    template <class T> T &KKm(int m, T &res, T &logres, int &logresvalid, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *s, int justcalcip, int indstart, int indend, const Vector<int> *iset, int skipbias = 0, int skipx = 0) const;

    template <class T> T &LL0(T &res, T &logres, int &logresvalid, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip, int indstart, int indend) const;
    template <class T> T &LL1(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const vecInfo &xainfo, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int indstart, int indend) const;
    template <class T> T &LL2(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int justcalcip, int indstart, int indend) const;
    template <class T> T &LL3(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const Vector<int> *s, int justcalcip, int indstart, int indend) const;
    template <class T> T &LL4(T &res, T &logres, int &logresvalid, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int ic, int id, int xdim, int xconsist, int assumreal, int xresmode, int mlid, const Vector<int> *s, int justcalcip, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int indstart, int indend) const;
    template <class T> T &LLm(int m, T &res, T &logres, int &logresvalid, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, const Matrix<double> *xy, const Vector<int> *s, int justcalcip, int indstart, int indend) const;








    template <class T> void yyydKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void yyyd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void yyydnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;

    template <class T> void yyyadKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void yyyad2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void yyyadnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;

    template <class T> void yyybdKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void yyybd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void yyybdnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;

    template <class T> void yyycdKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void yyycd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void yyycdnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;

    template <class T> void qqqdK2delx(T &xscaleres, T &yscaleres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int iaset, int ibset, int assumreal) const;
    template <class T> void qqqd2K2delxdelx(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDerive, int iaset, int ibset, int assumreal) const;
    template <class T> void qqqd2K2delxdely(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDerive, int iaset, int ibset, int assumreal) const;
    template <class T> void qqqdnK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDerive, int iaset, int ibset, int assumreal) const;

    template <class T> void xdKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void xd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;
    template <class T> void xdnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iset, int jset) const;

    template <class T> void dKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset, int skipbias = 0, int skipxa = 0, int skipxb = 0) const;
    template <class T> void d2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset, int skipbias = 0, int skipxa = 0, int skipxb = 0) const;
    template <class T> void dnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset, int skipbias = 0, int skipxa = 0, int skipxb = 0) const;

    template <class T> void dLL2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void d2LL2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;
    template <class T> void dnLL2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const;




    template <class T> int  KKpro(  T &res, const T &xyprod, const T &diffis, int *i, int locindstart, int locindend, int xdim, int m, T &logres, const T *xprod) const;
    template <class T> void dKKpro( T &xygrad, T &xnormgrad, T &res, const T &xyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, const T &xxprod, const T &yyprod) const;
    template <class T> void d2KKpro(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, T &res, const T &xyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, const T &xxprod, const T &yyprod) const;
    template <class T> void dnKKpro(T &res, const Vector<int> &gd, const T &xyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, int isfirstcalc, T &scratch) const;


    template <class T> int  KKprosingle(  T &res, const T &xyprod, const T &diffis, int *i, int xdim, int m, T &logres, const T *xprod, int ktype, int &logresvalid, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const;
    template <class T> void dKKprosingle( T &xygrad, T &diffgrad, T &xnormonlygrad, T &res, const T &xyprod, const T &diffis, int i, int j, int xdim, int m, const T &xxprod, const T &yyprod, int ktype, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const;
    template <class T> void d2KKprosingle(T &xygrad, T &diffgrad, T &xnormonlygrad, T &xyxygrad, T &diffdiffgrad, T &xnormxnormonlygrad, T &xnormynormonlygrad, T &ynormynormonlygrad, T &res, const T &xyprod, const T &diffis, int i, int j, int xdim, int m, const T &xxprod, const T &yyprod, int ktype, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const;
//    template <class T> void dnKKprosingle(T &res, const Vector<int> &gd, const T &xyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, int isfirstcalc, T &scratch, const T &xxprod, const T &yyprod, int ktype, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const;


    void K0i(     gentype &res,        const gentype &xyprod,                                    int xdim, int densetype, int resmode, int mlid, int indstart, int indend) const;
    void K0(      gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, int xdim, int densetype, int resmode, int mlid) const;
    void K0unnorm(gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, int xdim, int densetype, int resmode, int mlid) const;


    void K2i(     gentype &res,        const gentype &xyprod, const gentype &yxprod,                                    const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int densetype, int resmode, int mlid, int indstart, int indend, int assumreal) const;
    void K2(      gentype &res, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int densetype, int resmode, int mlid) const;
    void K2unnorm(gentype &res, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int densetype, int resmode, int mlid) const;


    void K4i(     gentype &res,        const gentype &xyprod,                                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, int i, int j, int k, int l, int xdim, int densetype, int resmode, int mlid, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s, int indstart, int indend, int assumreal) const;
    void K4(      gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, int i, int j, int k, int l, int xdim, int densetype, int resmode, int mlid, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const;
    void K4unnorm(gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, int i, int j, int k, int l, int xdim, int densetype, int resmode, int mlid, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const;


    void Kmi(     gentype &res,        const gentype &xyprod,                                    Vector<const vecInfo *> &xinfo, Vector<const gentype *> &xnorm, Vector<const SparseVector<gentype> *> &x, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid, const Matrix<double> &xy, const Vector<int> *s, int indstart, int indend, int assumreal) const;
    void Km(      gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, Vector<const vecInfo *> &xinfo, Vector<const gentype *> &xnorm, Vector<const SparseVector<gentype> *> &x, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid, const Matrix<double> &xy, const Vector<int> *s) const;
    void Kmunnorm(gentype &res, int q, const gentype &xyprod, gentype &diffis, int recalcdiffis, Vector<const vecInfo *> &xinfo, Vector<const gentype *> &xnorm, Vector<const SparseVector<gentype> *> &x, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid, const Matrix<double> &xy, const Vector<int> *s) const;


    void Kbase(gentype &res, int q, int typeis,
               const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
               Vector<const SparseVector<gentype> *> &x,
               Vector<const vecInfo *> &xinfo,
               Vector<const gentype *> &xnorm,
               Vector<int> &i,
               int xdim, int m, int densetype, int resmode, int mlid) const;

    // Kernel normalisation constants (altdiff 2,3)

    double AltDiffNormConst(int xdim, int m, double gamma) const
    {
        return ( ( m == 0 ) || ( m == 2 ) || ( !xdim ) || ( isAltDiff() != 2 ) ) ? 1 : (pow(2.0/m,xdim/2.0)*pow(2.0/(NUMBASE_PI*gamma*gamma),(xdim/2.0)*((m/2.0)-1)));
    }

    // Derivative tree - first order
    //
    // in dKmdx: x is the first argument, "y" is the elementwise product of all other arguments

    void dKdaz(gentype &resda, gentype &resdz, int &minmaxind, const gentype &xyprod, const gentype &yxprod, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid, int assumreal) const;

    void dKda(gentype &res, int &minmaxind, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid) const;
    void dKdz(gentype &res, int &minmaxind, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid) const;

    void dKunnormda(gentype &res, int &minmaxind, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid) const;
    void dKunnormdz(gentype &res, int &minmaxind, int q, const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int i, int j, int xdim, int mlid) const;


    void dKdaBase(gentype &res, int &minmaxind, int q, 
                  const gentype &xyprod, const gentype &yxprod, const gentype &diffis, 
                  Vector<const SparseVector<gentype> *> &x,
                  Vector<const vecInfo *> &xinfo,
                  Vector<const gentype *> &xnorm,
                  Vector<int> &i,
                  int xdim, int m, int mlid) const;

    void dKdzBase(gentype &res, int &minmaxind, int q, 
                  const gentype &xyprod, const gentype &yxprod, const gentype &diffis, 
                  Vector<const SparseVector<gentype> *> &x,
                  Vector<const vecInfo *> &xinfo,
                  Vector<const gentype *> &xnorm,
                  Vector<int> &i,
                  int xdim, int m, int mlid) const;


    // Kernel fall-through for 800-series kernels

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa,
                   const vecInfo &xainfo,
                   int ia, 
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                   int ia, int ib, int ic,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                   int ia, int ib, int ic, int id,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                   Vector<const SparseVector<gentype> *> &x,
                   Vector<const vecInfo *> &xinfo,
                   Vector<int> &i,
                   int xdim, int m, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, 
                   const vecInfo &xainfo, 
                   int ia, 
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                   int ia, int ib, int ic, 
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                   int ia, int ib, int ic, int id,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel8xx(int q, double &res, int &minmaxind, int typeis,
                   Vector<const SparseVector<gentype> *> &x,
                   Vector<const vecInfo *> &xinfo,
                   Vector<int> &i,
                   int xdim, int m, int densetype, int resmode, int mlid) const;

    void dkernel8xx(int q, gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int densetype, int resmode, int mlid) const;

    void dkernel8xx(int q, double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int densetype, int resmode, int mlid) const;


    // Kernel seek for 9xx kernels

    void kernel9xx(int q, gentype &res, int &minmaxind, int typeis,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, gentype &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa,
                   const vecInfo &xainfo,
                   int ia, 
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, gentype &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, gentype &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                   int ia, int ib, int ic,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, gentype &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                   int ia, int ib, int ic, int id,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, gentype &res, int &minmaxind, int typeis,
                   Vector<const SparseVector<gentype> *> &x,
                   Vector<const vecInfo *> &xinfo,
                   Vector<int> &i,
                   int xdim, int m, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, double &res, int &minmaxind, int typeis,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, double &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, 
                   const vecInfo &xainfo, 
                   int ia, 
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, double &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                   const vecInfo &xainfo, const vecInfo &xbinfo,
                   int ia, int ib,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, double &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                   int ia, int ib, int ic, 
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, double &res, int &minmaxind, int typeis,
                   const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                   const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                   int ia, int ib, int ic, int id,
                   int xdim, int densetype, int resmode, int mlid) const;

    void kernel9xx(int q, double &res, int &minmaxind, int typeis,
                   Vector<const SparseVector<gentype> *> &x,
                   Vector<const vecInfo *> &xinfo,
                   Vector<int> &i,
                   int xdim, int m, int densetype, int resmode, int mlid) const;




    // Final destination inner products, where redirection occurs from if
    // redirction is turned on.
    //
    // inding: 0 - not indexed
    //         1 - indexed
    // conj: 0 - no conj
    //       1 - normal conj operation
    //       2 - reversed conj
    // scaling: 0 - no scale
    //          1 - left scale
    //          2 - right scale
    //          3 - left/right scale
    //
    // Return value: 0 if result is not an equation or distribution (uses isValEqn)
    //               nz otherwise (if res has type != gentype then result *not* set)

    int twoProductDivertedNoConj (gentype &res,                       const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const;
    int twoProductDivertedRevConj(gentype &res, const gentype &xyres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const;

    int twoProductDivertedNoConj (double  &res,                       const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const { return twoProductDiverted(res,a,b,xconsist,assumreal); }
    int twoProductDivertedRevConj(double  &res, const double  &xyres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const { (void) a; (void) b; (void) xconsist; (void) assumreal; res = xyres; return 0; }

    int oneProductDiverted  (       gentype &res, const SparseVector<gentype> &a, int xconsist = 0, int assumreal = 0) const;
    int twoProductDiverted  (       gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const;
    int threeProductDiverted(       gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, int xconsist = 0, int assumreal = 0) const;
    int fourProductDiverted (       gentype &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d, int xconsist = 0, int assumreal = 0) const;
    int mProductDiverted    (int m, gentype &res, const Vector<const SparseVector<gentype> *> &a, int xconsist = 0, int assumreal = 0) const;

    int oneProductDiverted  (       double &res, const SparseVector<gentype> &a, int xconsist = 0, int assumreal = 0) const { gentype temp(res); int tres = oneProductDiverted(temp,a,xconsist,assumreal); res = tres ? 0.0 : (double) temp; return tres; }
    int twoProductDiverted  (       double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist = 0, int assumreal = 0) const { gentype temp(res); int tres = twoProductDiverted(temp,a,b,xconsist,assumreal); res = tres ? 0.0 : (double) temp; return tres; }
    int threeProductDiverted(       double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, int xconsist = 0, int assumreal = 0) const { gentype temp(res); int tres = threeProductDiverted(temp,a,b,c,xconsist,assumreal); res = tres ? 0.0 : (double) temp; return tres; }
    int fourProductDiverted (       double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d, int xconsist = 0, int assumreal = 0) const { gentype temp(res); int tres = fourProductDiverted(temp,a,b,c,d,xconsist,assumreal); res = tres ? 0.0 : (double) temp; return tres; }
    int mProductDiverted    (int m, double &res, const Vector<const SparseVector<gentype> *> &a, int xconsist = 0, int assumreal = 0) const { gentype temp(res); int tres = mProductDiverted(m,temp,a,xconsist,assumreal); res = tres ? 0.0 : (double) temp; return tres; }

    // Further in, deeper down

    void getOneProd  (gentype &res, const SparseVector<gentype> &xa, int inding, int scaling, int xconsist, int assumreal) const;
    void getTwoProd  (gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, int inding, int conj, int scaling, int xconsist, int assumreal) const;
    void getThreeProd(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, int inding, int scaling, int xconsist, int assumreal) const;
    void getFourProd (gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, int inding, int scaling, int xconsist, int assumreal) const;
    void getmProd    (gentype &res, const Vector<const SparseVector<gentype> *> &x, int inding, int scaling, int xconsist, int assumreal) const;

    void fixShiftProd(void);

    SparseVector<gentype> &preShiftScale(SparseVector<gentype> &res, const SparseVector<gentype> &x) const;

    // Note: diff0norm and diff1norm evaluate to zero in all cases, so we bypass them here

           void diff0norm(gentype &res, const gentype &xyprod) const { (void) xyprod; res = 0; return; }
           void diff1norm(gentype &res, const gentype &xyprod, const gentype &xanorm) const { (void) xyprod; (void) xanorm; res = 0; return; }
           void diff2norm(gentype &res, const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm) const { res = xyprod; res *= -2.0; res += xanorm; setconj(res); res += xbnorm; setconj(res); return; }
    inline void diff3norm(gentype &res, const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s = NULL) const;
    inline void diff4norm(gentype &res, const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s = NULL) const;
    inline void diffmnorm(int m, gentype &res, const gentype &xyprod, const Vector<const gentype *> &xanorm, const Matrix<double> &xy, const Vector<int> *s = NULL) const;

           void diff0norm(double &res, const double &xyprod) const { (void) xyprod; res = 0; return; }
           void diff1norm(double &res, const double &xyprod, const double &xanorm) const { (void) xyprod; (void) xanorm; res = 0; return; }
           void diff2norm(double &res, const double &xyprod, const double &xanorm, const double &xbnorm) const { res = xanorm+xbnorm-(2*xyprod); return; }
    inline void diff3norm(double &res, const double &xyprod, const double &xanorm, const double &xbnorm, const double &xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s = NULL) const;
    inline void diff4norm(double &res, const double &xyprod, const double &xanorm, const double &xbnorm, const double &xcnorm, const double &xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s = NULL) const;
    inline void diffmnorm(int m, double &res, const double &xyprod, const Vector<const double *> &xanorm, const Matrix<double> &xy, const Vector<int> *s = NULL) const;

    void diff0norm(gentype &res, const double &xyprod) const { diff0norm(res.force_double(),xyprod); return; }
    void diff1norm(gentype &res, const double &xyprod, const double &xanorm) const { diff1norm(res.force_double(),xyprod,xanorm); return; }
    void diff2norm(gentype &res, const double &xyprod, const double &xanorm, const double &xbnorm) const { diff2norm(res.force_double(),xyprod,xanorm,xbnorm); return; }
    void diff3norm(gentype &res, const double &xyprod, const double &xanorm, const double &xbnorm, const double &xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s = NULL) const { diff3norm(res.force_double(),xyprod,xanorm,xbnorm,xcnorm,xy00,xy10,xy11,xy20,xy21,xy22,s); return; }
    void diff4norm(gentype &res, const double &xyprod, const double &xanorm, const double &xbnorm, const double &xcnorm, const double &xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s = NULL) const { diff4norm(res.force_double(),xyprod,xanorm,xbnorm,xcnorm,xdnorm,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s); return; }
    void diffmnorm(int m, gentype &res, const double &xyprod, const Vector<const double *> &xanorm, const Matrix<double> &xy, const Vector<int> *s = NULL) const { diffmnorm(m,res.force_double(),xyprod,xanorm,xy,s); return; }

    // If optionCache set then dereferences this, otherwise resizes altres to m*m and fills it will <x,y> products.

    void fillXYMatrix(double &altxyr00, double &altxyr10, double &altxyr11, double &altxyr20, double &altxyr21, double &altxyr22, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int doanyhow = 0, int assumreal = 0) const;
    void fillXYMatrix(double &altxyr00, double &altxyr10, double &altxyr11, double &altxyr20, double &altxyr21, double &altxyr22, double &altxyr30, double &altxyr31, double &altxyr32, double &altxyr33, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int doanyhow = 0, int assumreal = 0) const;

    const Matrix<double> &fillXYMatrix(int m, Matrix<double> &altres, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const Matrix<double> *optionCache = NULL, int doanyhow = 0, int assumreal = 0) const;

    // Function to overwrite constants (integer and real) from source vectors

    Vector<int> combinedOverwriteSrc;

    int backupisind; // don't worry that this is not initialised - it is
                     // always set when required.  Also no need to save this.
    Vector<int> backupdIndexes;

    void processOverwrites(int q, const SparseVector<gentype> &x, const SparseVector<gentype> &y) const;
    void fixcombinedOverwriteSrc(void);
    void addinOverwriteInd(const SparseVector<gentype> &x, const SparseVector<gentype> &y) const;
    void addinOverwriteInd(const Vector<gentype> &x, const Vector<gentype> &y) const;
//    void addinOverwriteInd(const SparseVector<double> &x, const SparseVector<double> &y) const;
//    void addinOverwriteInd(const Vector<double> &x, const Vector<double> &y) const;
    void addinOverwriteInd(const SparseVector<gentype> &v) const;
    void addinOverwriteInd(const SparseVector<gentype> &v, const SparseVector<gentype> &w, const SparseVector<gentype> &x) const;
    void addinOverwriteInd(const SparseVector<gentype> &v, const SparseVector<gentype> &w, const SparseVector<gentype> &x, const SparseVector<gentype> &y) const;
    void addinOverwriteInd(const Vector<const SparseVector<gentype> *> &a) const;
    void addinOverwriteInd(void) const;
    void removeOverwriteInd(void) const;

    int arexysimple(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd) const
    {
        if ( !arexysimple(xa,xb) )
        {
            return 0;
        }

        if ( !arexysimple(xa,xc) )
        {
            return 0;
        }

        if ( !arexysimple(xa,xd) )
        {
            return 0;
        }

        return 1;
    }

    int arexysimple(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc) const
    {
        if ( !arexysimple(xa,xb) )
        {
            return 0;
        }

        if ( !arexysimple(xa,xc) )
        {
            return 0;
        }

        return 1;
    }

    int arexysimple(const SparseVector<gentype> &x, const SparseVector<gentype> &y) const
    {
        if ( ( x.nearindsize() == 0 ) && ( y.nearindsize() == 0 ) )
        {
            return 1;
        }

        if ( ( x.nearindsize() == 1 ) && ( y.nearindsize() == 1 ) )
        {
            if ( x.ind(0) == y.ind(0) )
            {
                return 1;
            }
        }

        return 0;
    }

    int arexysimple(const SparseVector<gentype> &x) const
    {
        if ( ( x.nearindsize() == 0 ) || ( x.nearindsize() == 1 ) )
        {
             return 1;
        }

        return 0;
    }

    int arexysimple(int m, const Vector<const SparseVector<gentype> *> &x) const
    {
        NiceAssert( m <= x.size() );

        if ( m > 1 )
        {
            int i = 0;

            for ( i = 1 ; i < m ; i++ )
            {
                if ( !arexysimple(*(x(zeroint())),*(x(i))) )
                {
                    return 0;
                }
            }
        }

        return 1;
    }

    // The only relevant part of indres is the index vector

    void combind(SparseVector<gentype> &indres, const SparseVector<gentype> &x, const SparseVector<gentype> &y) const
    {
        indres.resize(0);

        if ( x.nearindsize() )
        {
            int i;

            for ( i = 0 ; i < x.nearindsize() ; i++ )
            {
                indres("&",x.ind(i)) = x.direcref(i);
            }
        }

        if ( y.nearindsize() )
        {
            int i;

            for ( i = 0 ; i < y.nearindsize() ; i++ )
            {
                indres("&",y.ind(i)) = y.direcref(i);
            }
        }

        return;
    }

    void combind(int m, SparseVector<gentype> &indres, const Vector<const SparseVector<gentype> *> &x) const
    {
        indres.resize(0);

        int i,j;

        if ( m )
        {
            for ( j = 0 ; j < m ; j++ )
            {
                if ( (*(x(j))).nearindsize() )
                {
                    for ( i = 0 ; i < (*(x(j))).nearindsize() ; i++ )
                    {
                        indres("&",(*(x(j))).ind(i)) = (*(x(j))).direcref(i);
                    }
                }
            }
        }

        return;
    }

    // Given the result of kernel evaluation, this function will attempt
    // to calculate xyprod and yxprod.  Returns 0 on success, nz on fail.

    int reverseEngK(gentype &res, const vecInfo &xinfo, const vecInfo &yinfo, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const double &Kres) const;
    int reverseEngK(gentype &res, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const double &Kres) const;
    int reverseEngK(int m, gentype &res, const Vector<const vecInfo *> &xinfo, const Vector<const SparseVector<gentype> *> &x, const double &Kres) const;

    int isKernelDerivativeEasy(void) const
    {
//        return !isProd() &&
//               !isIndex() &&
//               !isShifted() &&
//               !isScaled() &&
//               ( size() == 1 ) &&
//               !isNormalised(0) &&
//               ( kinf(0).numflagsset() <= 1 ) &&
//               ( ( kinf(0).numflagsset() == 0 ) || kinf(0).usesDiff || kinf(0).usesInner || kinf(0).usesMinDiff || kinf(0).usesMaxDiff );
        return !isProd() &&
               ( size() == 1 ) &&
               !isNormalised(0) &&
               ( kinf(0).numflagsset() <= 1 ) &&
               ( ( kinf(0).numflagsset() == 0 ) || kinf(0).usesDiff || kinf(0).usesInner || kinf(0).usesMinDiff || kinf(0).usesMaxDiff );
    }

    // Function to tell us if kerrnel index actually exists

    int iskern(int potind) const;

    // This function returns true if, despite redirected inner products, the
    // kernel will still require the actual vectors themselves.  Note that
    // evaluation may still throw an exception if overwrites must be processed
    // or pre-processing on vector basis is requested.

    int needExplicitVector(void) const
    {
        return kinf(0).usesVector || isProd();
    }

    MercerKernel *thisthis;
    MercerKernel **thisthisthis;

    // combine minmaxadd

    int combineminmaxind(int aminmaxind, int bminmaxind) const
    {
        if ( aminmaxind == -2 )
        {
            return ( bminmaxind == -2 ) ? -1 : bminmaxind;
        }

        if ( bminmaxind == -2 )
        {
            return ( aminmaxind == -2 ) ? -1 : aminmaxind;
        }

        if ( ( aminmaxind == -1 ) && ( bminmaxind == -1 ) )
        {
            return -1;
        }

        throw("Incompatible gradient indices error");

        return -3;
    }











    // xisfast: -1 if unknown
    //          0  if kernel is not fast and full calculation is required
    //          1  if completely unchained kernel where all kernels are either inner-product or diff
    //          2  if completely chained kernel where kernel 0 is either inner-product or diff, and remaining kernels are inner-product (no splits allowed or magterms)
    //          3  if completely chained kernel where kernel 0 is kernel transfer, kernel 1 is either inner-product or diff, and remaining kernels are inner-product (no splits or magterms allowed)
    // xneedsInner: needs inner product to calculate
    // xneedsDiff:  needs inner product to calculate
    // xneedsNorm:  needs norms of vectors to calculate

    int xisfast;
    int xneedsInner;
    int xneedsDiff;
    int xneedsNorm;

    // needsInner:   returns 1 if inner (m) product is required in this kernel (-1 for all parts)
    // needsDiff:    returns 1 if diff (m) product is required in this kernel (-1 for all parts)
    // needsMatDiff: returns 1 if diff (m) product is required and matrix accelerated works.
    // needsNorm:    returns 1 if ubber (m) norm(s) is(are) required in this kernel (-1 for all parts)

    int needsMatDiff(int q = -1) const 
    { 
        return needsDiff(q) && ( ( isAltDiff() == 2   ) || ( isAltDiff() == 3   ) || ( isAltDiff() == 4   ) || 
                                 ( isAltDiff() == 102 ) || ( isAltDiff() == 103 ) || ( isAltDiff() == 104 ) || 
                                 ( isAltDiff() == 202 ) || ( isAltDiff() == 203 ) || ( isAltDiff() == 204 ) ||
                                 ( isAltDiff() == 300 )    );
    }

    int isFastKernelType(int ind) const
    {
        return ( ( cType(ind) == 0   ) ||
                 ( cType(ind) == 1   ) ||
                 ( cType(ind) == 2   ) ||
                 ( cType(ind) == 3   ) ||
                 ( cType(ind) == 4   ) ||
                 ( cType(ind) == 5   ) ||
                 ( cType(ind) == 7   ) ||
                 ( cType(ind) == 8   ) ||
                 ( cType(ind) == 9   ) ||
                 ( cType(ind) == 10  ) ||
                 ( cType(ind) == 11  ) ||
                 ( cType(ind) == 12  ) ||
                 ( cType(ind) == 13  ) ||
                 ( cType(ind) == 14  ) ||
                 ( cType(ind) == 15  ) ||
                 ( cType(ind) == 19  ) ||
                 ( cType(ind) == 23  ) ||
                 ( cType(ind) == 24  ) ||
                 ( cType(ind) == 25  ) ||
                 ( cType(ind) == 26  ) ||
                 ( cType(ind) == 27  ) ||
                 ( cType(ind) == 32  ) ||
                 ( cType(ind) == 33  ) ||
                 ( cType(ind) == 34  ) ||
                 ( cType(ind) == 38  ) ||
                 ( cType(ind) == 39  ) ||
                 ( cType(ind) == 42  ) ||
                 ( cType(ind) == 43  ) ||
                 ( cType(ind) == 44  ) ||
                 ( cType(ind) == 45  ) ||
                 ( cType(ind) == 46  ) ||
                 ( cType(ind) == 47  ) ||
                 ( cType(ind) == 100 ) ||
                 ( cType(ind) == 103 ) ||
                 ( cType(ind) == 104 ) ||
                 ( cType(ind) == 106 ) ||
                 ( cType(ind) == 200 ) ||
                 ( cType(ind) == 203 ) ||
                 ( cType(ind) == 204 ) ||
                 ( cType(ind) == 206 )    );
    }

    int isfast(void) const
    {
        int res = xisfast;

        if ( xisfast == -1 )
        {
             res = (**thisthisthis).isfastunsafe();
        }

        return res;
    }

    int needsInner(int q = -1, int m = 2) const
    {
        NiceAssert( q < size() );
        NiceAssert( q >= -1 );

        int res = xneedsInner;

        if ( q >= 0 )
        {
            res = kinf(q).usesInner || ( kinf(q).usesDiff && ( ( m == 2 ) || ( isAltDiff() <= 1 ) ) );
        }

        else if ( xneedsInner == -1 )
        {
            res = (**thisthisthis).needsInnerunsafe(m);
        }

        return res;
    }

    int needsDiff(int q = -1) const
    {
        NiceAssert( q < size() );
        NiceAssert( q >= -1 );

        int res = xneedsDiff;

        if ( q >= 0 )
        {
            res = kinf(q).usesDiff;
        }

        else if ( xneedsDiff == -1 )
        {
            res = (**thisthisthis).needsDiffunsafe();
        }

        return res;
    }

    int needsNorm(int q = -1) const
    {
        NiceAssert( q < size() );
        NiceAssert( q >= -1 );

        int res = xneedsNorm;

        if ( q >= 0 )
        {
            res = needsDiff(q) || ( needsInner(q) && isMagTerm(q) );
        }

        else if ( xneedsNorm == -1 )
        {
            res = (**thisthisthis).needsNormunsafe();
        }

        return res;
    }

    int isfastunsafe(void)
    {
        svmvolatile static svm_mutex eyelock;
        svm_mutex_lock(eyelock);

        // xisfast: -1 if unknown

        if ( xisfast == -1 )
        {
            xisfast = 0;

            //          0  if kernel is not fast and full calculation is required

            retVector<int> tmpva;

            if ( ( size() <= 1 ) || ( isnorm(zeroint(),1,size()-2,tmpva) == zeroint() ) )
            {
                if ( ( size() >= 1 ) && ( isnorm(size()-1) == zeroint() ) && ( ischain(zeroint(),1,size()-2,tmpva) == zeroint() ) )
                {
                    // Could be xisfast == 1
                    //          1  if completely unchained kernel where all kernels are either inner-product or diff

                    xisfast = 1;

                    int i;

                    for ( i = 0 ; i < size() ; i++ )
                    {
                        if ( !isFastKernelType(i) )
                        {
                            xisfast = 0;
                            break;
                        }
                    }
                }

                else if ( !numSplits() && !numMulSplits() && ( ( size() >= 1 ) && ( ( isnorm(size()-1) == zeroint() ) || ( isAltDiff() <= 99 ) ) && ( ( ( size() == 1 ) || ( ( ischain(zeroint(),1,size()-2,tmpva) == 1 ) ) ) && ( cType(0) < 800 ) ) ) )
                {
                    // Could be xisfast == 2
                    //          2  if completely chained kernel where kernel 0 is either inner-product or diff, and remaining kernels are inner-product (no splits allowed or magterms)

                    xisfast = 2;

                    int i;

                    for ( i = 0 ; i < size() ; i++ )
                    {
                        if ( !isFastKernelType(i) || ( i && needsDiff(i) ) || isMagTerm(i) )
                        {
                            xisfast = 0;
                            break;
                        }
                    }
                }

                else if ( !numSplits() && !numMulSplits() && (  ( size() >= 1 ) && ( ( isnorm(size()-1) == zeroint() ) || ( isAltDiff() <= 99 ) ) && ( ( ( size() == 1 ) || ( ( ischain(zeroint(),1,size()-2,tmpva) == 1 ) ) ) && ( cType(0) >= 800 ) && ( cType(0) <= 999 ) ) ) )
                {
                    // Could be xisfast == 3
                    //          3  if completely chained kernel where kernel 0 is kernel transfer, kernel 1 is either inner-product or diff, and remaining kernels are inner-product (no splits or magterms allowed)

                    xisfast = 3;

                    int i;

                    for ( i = 0 ; i < size() ; i++ )
                    {
                        if ( ( i && ( !isFastKernelType(i) || ( ( i > 1 ) && needsDiff(i) ) ) ) || isMagTerm(i) )
                        {
                            xisfast = 0;
                            break;
                        }
                    }
                }
            }
        }

        svm_mutex_unlock(eyelock);

        return xisfast;
    }

    int needsInnerunsafe(int m)
    {
        svmvolatile static svm_mutex eyelock;
        svm_mutex_lock(eyelock);

        if ( xneedsInner == -1 )
        {
            int usesInner = 0;
            int usesDiff  = 0;

            if ( size() )
            {
                int q;

                for ( q = 0 ; q < size() ; q++ )
                {
                    usesInner |= kinf(q).usesInner;
                    usesDiff  |= kinf(q).usesDiff;
                }
            }

            xneedsInner = ( usesInner || ( usesDiff && ( ( m == 2 ) || ( isAltDiff() <= 1 ) ) ) ) ? 1 : 0;
        }

        svm_mutex_unlock(eyelock);

        return xneedsInner;
    }

    int needsDiffunsafe(void)
    {
        svmvolatile static svm_mutex eyelock;
        svm_mutex_lock(eyelock);

        if ( xneedsDiff == -1 )
        {
            int usesDiff  = 0;

            if ( size() )
            {
                int q;

                for ( q = 0 ; q < size() ; q++ )
                {
                    usesDiff  |= kinf(q).usesDiff;
                }
            }

            xneedsDiff = usesDiff ? 1 : 0;
        }

        svm_mutex_unlock(eyelock);

        return xneedsDiff;
    }

    int needsNormunsafe(void)
    {
        svmvolatile static svm_mutex eyelock;
        svm_mutex_lock(eyelock);

        if ( xneedsNorm == -1 )
        {
            int usesNorm = 0;

            if ( size() )
            {
                int q;

                for ( q = 0 ; q < size() ; q++ )
                {
                    usesNorm |= ( needsDiff(q) || ( needsInner(q) && isMagTerm(q) ) );
                }
            }

            xneedsNorm = usesNorm ? 1 : 0;
        }

        svm_mutex_unlock(eyelock);

        return xneedsNorm;
    }

    // Sampling functions for distribution kernels - return 0 if nothing is changed by sampling, >0 otherwise

    inline int subSample(SparseVector<SparseVector<gentype> > &subval, SparseVector<gentype> &x, vecInfo &xinfo) const;
    inline int subSample(SparseVector<SparseVector<gentype> > &subval, gentype &b) const;
    inline int subSample(SparseVector<SparseVector<gentype> > &subval, double  &b) const;

    // Various short-circuited kernels
    //
    // isSimpleKernel:           size 1, no normalisation, no chaining
    // isSimpleBasicKernel:      isSimpleKernel, and type   0-99  (NN kernel)
    // isSimpleNNKernel:         isSimpleKernel, and type 100-299 (NN kernel)
    // isSimpleDistKernel:       isSimpleKernel, and type 300-399 (-ve dist kernel)
    // isSimpleXferKernel:       isSimpleKernel, and type 800-999 (kernel transfer and socket)
    // isSimpleKernelChain:      size 2, no normalisation, chained, with kernel 0 being a kernel transfer

public:
    int isSimpleKernel     (void) const { return ( ( size() == 1 ) && !isNormalised() && !isChained() && !isSplit() && !isMulSplit() && !isMagTerm() ); }
    int isSimpleBasicKernel(void) const { return ( isSimpleKernel() && ( cType() >=   0 ) && ( cType() <  100 ) ); }
    int isSimpleNNKernel   (void) const { return ( isSimpleKernel() && ( cType() >= 100 ) && ( cType() <  300 ) ); }
    int isSimpleDistKernel (void) const { return ( isSimpleKernel() && ( cType() >= 300 ) && ( cType() <  400 ) ); }
    int isSimpleXferKernel (void) const { return ( isSimpleKernel() && ( cType() >= 800 ) && ( cType() <= 999 ) ); }
    int isSimpleKernelChain(void) const { return ( ( size() == 2 )  && ( cType() >= 800 ) && ( cType() <= 999 ) && !isNormalised(0) && !isNormalised(1) && isChained(0) && !isSplit(0) && !isMulSplit(0) && !isMagTerm() ); }
    int isTrivialKernel    (void) const { static gentype tempsampdist("[ ]"); return ( !isFullNorm() && !isProd() && !isIndex() && !isShifted() && !isScaled() && !isLeftPlain() && 
                                          !isRightPlain() && ( isAltDiff() == 1 ) && !isNormalised() && !isChained() && !isSplit() && !isMulSplit() && !isMagTerm() && ( size() == 1 ) && 
                                          ( numSamples() == DEFAULT_NUMKERNSAMP ) && ( sampleDistribution() == tempsampdist ) && ( sampleIndices().size() == 0 ) &&
                                          ( cRealOverwrite().indsize() == 0 ) && ( cIntOverwrite().indsize() == 0 ) ); }

    int isFastKernelSum  (void) const { return ( isfast() == 1 ); }
    int isFastKernelChain(void) const { return ( isfast() == 2 ); } // No splits or magnitude terms allowed
    int isFastKernelXfer (void) const { return ( isfast() == 3 ); } // No splits or magnitude terms allowed

    int isSimpleLinearKernel(void) const { return ( isSimpleKernel() && ( cType() == 1 ) ); }

private:
    // Kernel 900 stuff.  These are the unix sockets that are used to communicate
    // with an *external* kernel function.
    //
    // make900channel(q): make and return the unix socket stream for kernel q.
    //                    - if it exists already, return it
    //                    - if it doesn't exist but this isn't a 9xx kernel, return NULL.
    //                    - if it doesn't exist, is a 9xx kernel but index is non-positive, return NULL
    //                    - otherwise construct it, store it and return it.

    Vector<awarestream *> k900sock;

    awarestream *make900channel(int q) const
    {
        if ( k900sock(q) || ( dtype(q)/100 != 9 ) || ( dIntConstants(q)(0) < 0 ) )
        {
            return (**thisthisthis).k900sock(q);
        }

        int sn = dIntConstants(q)(0);

        char buffer[1024];

        sprintf(buffer,"%d",sn);

        std::string sockname("kern");
        sockname += buffer;
        sockname += ".sock";

        std::string serverurl("");

        awarestream *svmsock = NULL;

        // 900,901,902: unix socket server, construct right now
        // 903,904,905: unix socket client, delayed connection
        // 906,907,908: unix TCPIP server, construct right now
        // 909,910,911: unix socket client, delayed connection

             if ( dtype(q) == 900 ) { svmsock = makeUnixSocket(sockname,1,1,1); }
        else if ( dtype(q) == 901 ) { svmsock = makeUnixSocket(sockname,1,1,1); }
        else if ( dtype(q) == 902 ) { svmsock = makeUnixSocket(sockname,1,1,1); }
        else if ( dtype(q) == 903 ) { svmsock = makeUnixSocket(sockname,1,1,0); }
        else if ( dtype(q) == 904 ) { svmsock = makeUnixSocket(sockname,1,1,0); }
        else if ( dtype(q) == 905 ) { svmsock = makeUnixSocket(sockname,1,1,0); }
        else if ( dtype(q) == 906 ) { svmsock = makeTCPIPSocket(serverurl,sn,1,1); }
        else if ( dtype(q) == 907 ) { svmsock = makeTCPIPSocket(serverurl,sn,1,1); }
        else if ( dtype(q) == 908 ) { svmsock = makeTCPIPSocket(serverurl,sn,1,1); }
        else if ( dtype(q) == 909 ) { svmsock = makeTCPIPSocket(serverurl,sn,1,0); }
        else if ( dtype(q) == 910 ) { svmsock = makeTCPIPSocket(serverurl,sn,1,0); }
        else if ( dtype(q) == 911 ) { svmsock = makeTCPIPSocket(serverurl,sn,1,0); }

        (**thisthisthis).k900sock("&",q) = svmsock;

        return svmsock;
    }

    void kill900channel(int q = -1)
    {
        if ( q >= 0 )
        {
            if ( k900sock(q) )
            {
                {
                    std::ostream svmsockout(k900sock(q));

                    svmsockout << "stop\n";
                }

                delUnixSocket(k900sock("&",q));

                k900sock("&",q) = NULL;
            }
        }

        else
        {
            for ( q = 0 ; q < size() ; q++ )
            {
                kill900channel(q);
            }
        }

        return;
    }
};

inline void qswap(MercerKernel &a, MercerKernel &b)
{
    qswap(a.isprod              ,b.isprod              );
    qswap(a.isind               ,b.isind               );
    qswap(a.isfullnorm          ,b.isfullnorm          );
    qswap(a.isshift             ,b.isshift             );
    qswap(a.leftplain           ,b.leftplain           );
    qswap(a.rightplain          ,b.rightplain          );
    qswap(a.isdiffalt           ,b.isdiffalt           );
    qswap(a.xproddepth          ,b.xproddepth          );
    qswap(a.enchurn             ,b.enchurn             );
    qswap(a.xsuggestXYcache     ,b.xsuggestXYcache     );
    qswap(a.xisIPdiffered       ,b.xisIPdiffered       );
    qswap(a.xnumSplits          ,b.xnumSplits          );
    qswap(a.xnumMulSplits       ,b.xnumMulSplits       );

    qswap(a.dtype               ,b.dtype               );
    qswap(a.isnorm              ,b.isnorm              );
    qswap(a.ischain             ,b.ischain             );
    qswap(a.issplit             ,b.issplit             );
    qswap(a.mulsplit            ,b.mulsplit            );
    qswap(a.ismagterm           ,b.ismagterm           );
    qswap(a.dIndexes            ,b.dIndexes            );
    qswap(a.kernflags           ,b.kernflags           );
    qswap(a.dRealConstants      ,b.dRealConstants      );
    qswap(a.dIntConstants       ,b.dIntConstants       );
    qswap(a.dRealOverwrite      ,b.dRealOverwrite      );
    qswap(a.dIntOverwrite       ,b.dIntOverwrite       );
    qswap(a.altcallback         ,b.altcallback         );

    qswap(a.dShift              ,b.dShift              );
    qswap(a.dScale              ,b.dScale              );
    qswap(a.dShiftProd          ,b.dShiftProd          );
    qswap(a.dShiftProdNoConj    ,b.dShiftProdNoConj    );
    qswap(a.dShiftProdRevConj   ,b.dShiftProdRevConj   );

    qswap(a.xnumsamples         ,b.xnumsamples         );
    qswap(a.xindsub             ,b.xindsub             );
    qswap(a.xsampdist           ,b.xsampdist           );

    qswap(a.combinedOverwriteSrc,b.combinedOverwriteSrc);
    qswap(a.backupisind         ,b.backupisind         );
    qswap(a.backupdIndexes      ,b.backupdIndexes      );

    qswap(a.xisfast             ,b.xisfast             );
    qswap(a.xneedsInner         ,b.xneedsInner         );
    qswap(a.xneedsDiff          ,b.xneedsDiff          );
    qswap(a.xneedsNorm          ,b.xneedsNorm          );

    qswap(a.k900sock,b.k900sock);

    return;
}





// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================

inline
const gentype &MercerKernel::getmnorm(const vecInfo &xinfo, const SparseVector<gentype> &x, int m, int xconsist, int assumreal) const
{
    m = isAltDiff() ? 2 : m;
    //m = isAltDiff() ? diffnormdefault() : m;

    if ( !(m%2) )
    {
        Vector<gentype> &xhalfmprod = **(xinfo.xhalfinda()); // strip the const

        int oldm = 2*(xhalfmprod.size());

        xhalfmprod.resize(m/2);

        if ( ( m >= 2 ) && ( m > oldm ) )
        {
            twoProductDivertedNoConj(xhalfmprod("&",zeroint()),x,x,xconsist,assumreal);

            oldm = 2;
        }

        if ( ( m >= 4 ) && ( m > oldm ) )
        {
            fourProductDiverted(xhalfmprod("&",1),x,x,x,x,xconsist,assumreal);

            oldm = 4;
        }

        if ( ( m >= 6 ) && ( m > oldm ) )
        {
            Vector<const SparseVector<gentype> *> aa(m);
            Vector<const vecInfo *> aainfo(m);

            aa     = &x;
            aainfo = &xinfo;

            int i;

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;

            for ( i = oldm+2 ; i <= m ; i += 2 )
            {
                mProductDiverted(i,xhalfmprod("&",(i/2)-1),aa(zeroint(),1,i-1,tmpva),xconsist,assumreal);
            }

            oldm = m;
        }

        return xhalfmprod((m/2)-1);
    }

    //FIXME: hack
    static Vector<gentype> scratch(20);
    static int scratchind = 19;

    scratchind++;
    scratchind %= 20;

    if ( m == 1 )
    {
        oneProductDiverted(scratch("&",scratchind),x,xconsist,assumreal);
    }

    else if ( m == 3 )
    {
        threeProductDiverted(scratch("&",scratchind),x,x,x,xconsist,assumreal);
    }

    else
    {
        Vector<const SparseVector<gentype> *> aa(m);
        Vector<const vecInfo *> aainfo(m);

        aa     = &x;
        aainfo = &xinfo;

        mProductDiverted(m,scratch("&",scratchind),aa,xconsist,assumreal);
    }
 
    return scratch(scratchind);
}

inline
gentype &MercerKernel::getmnorm(vecInfo &xinfo, const SparseVector<gentype> &x, int m, int xconsist, int assumreal) const
{
    m = isAltDiff() ? 2 : m;
    //m = isAltDiff() ? diffnormdefault() : m;

    if ( !(m%2) )
    {
        Vector<gentype> &xhalfmprod = **(xinfo.xhalfinda()); // strip the const

        int oldm = 2*(xhalfmprod.size());

        xhalfmprod.resize(m/2);

        if ( ( m >= 2 ) && ( m > oldm ) )
        {
            twoProductDivertedNoConj(xhalfmprod("&",zeroint()),x,x,xconsist,assumreal);

            oldm = 2;
        }

        if ( ( m >= 4 ) && ( m > oldm ) )
        {
            fourProductDiverted(xhalfmprod("&",1),x,x,x,x,xconsist,assumreal);

            oldm = 4;
        }

        if ( ( m >= 6 ) && ( m > oldm ) )
        {
            Vector<const SparseVector<gentype> *> aa(m);
            Vector<const vecInfo *> aainfo(m);

            aa     = &x;
            aainfo = &xinfo;

            int i;

            for ( i = oldm+2 ; i <= m ; i += 2 )
            {
                retVector<const SparseVector<gentype> *> tmpva;
                retVector<const vecInfo *>               tmpvb;

                mProductDiverted(i,xhalfmprod("&",(i/2)-1),aa(zeroint(),1,i-1,tmpva),xconsist,assumreal);
            }
        }

        return xhalfmprod("&",(m/2)-1);
    }

    //FIXME: hack
    static Vector<gentype> scratch(20);
    static int scratchind = 19;

    scratchind++;
    scratchind %= 20;

    if ( m == 1 )
    {
        oneProductDiverted(scratch("&",scratchind),x,xconsist,assumreal);
    }

    else if ( m == 3 )
    {
        threeProductDiverted(scratch("&",scratchind),x,x,x,xconsist,assumreal);
    }

    else
    {
        Vector<const SparseVector<gentype> *> aa(m);
        Vector<const vecInfo *> aainfo(m);

        aa     = &x;
        aainfo = &xinfo;

        mProductDiverted(m,scratch("&",scratchind),aa,xconsist,assumreal);
    }
 
    return scratch("&",scratchind);
}

inline
void MercerKernel::diff3norm(gentype &res, const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s) const
{
    (void) s;

    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    if ( isAltDiff() == 0 )
    {
        res  = xyprod;
        res *= -3.0;
        res += xanorm;
        res += xbnorm;
        res += xcnorm;
    }

    else if ( isAltDiff() == 1 )
    {
        res  = xyprod;
        res *= -2.0;
        res += xanorm;
        res += xbnorm;
        res += xcnorm;
    }

    else if ( isAltDiff() == 2 )
    {
        res  = xanorm;
        res += xbnorm;
        res += xcnorm;

        res -= ( xy00 + xy10 + xy20 )/3.0;
        res -= ( xy10 + xy11 + xy21 )/3.0;
        res -= ( xy20 + xy21 + xy22 )/3.0;

        res *= 2.0;
    }

    else
    {
        throw("diff3norm not defined for altdiff != 0,1,2");
    }

    return;
}

inline
void MercerKernel::diff4norm(gentype &res, const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const
{
    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    int z = 0;

    if ( isAltDiff() == 0 )
    {
        res  = xyprod;
        res *= -4.0;
        res += xanorm;
        res += xbnorm;
        res += xcnorm;
        res += xdnorm;
    }

    else if ( isAltDiff() == 1 )
    {
        res  = xyprod;
        res *= -2.0;
        res += xanorm;
        res += xbnorm;
        res += xcnorm;
        res += xdnorm;
    }

    else if ( isAltDiff() == 2 )
    {
        res  = xanorm;
        res += xbnorm;
        res += xcnorm;
        res += xdnorm;

        res -= ( xy00 + xy10 + xy20 + xy30)/4.0;
        res -= ( xy10 + xy11 + xy21 + xy31)/4.0;
        res -= ( xy20 + xy21 + xy22 + xy32)/4.0;
        res -= ( xy30 + xy31 + xy32 + xy33)/4.0;

        res *= 2.0;
    }

    else if ( isAltDiff() == 103 )
    {
        NiceAssert(s);

        res  = ((*s)(z)*(*s)(z)*xy00) + ((*s)(z)*(*s)(1)*xy10) + ((*s)(z)*(*s)(2)*xy20) + ((*s)(z)*(*s)(3)*xy30);
        res += ((*s)(1)*(*s)(z)*xy10) + ((*s)(1)*(*s)(1)*xy11) + ((*s)(1)*(*s)(2)*xy21) + ((*s)(1)*(*s)(3)*xy31);
        res += ((*s)(2)*(*s)(z)*xy20) + ((*s)(2)*(*s)(1)*xy21) + ((*s)(2)*(*s)(2)*xy22) + ((*s)(2)*(*s)(3)*xy32);
        res += ((*s)(3)*(*s)(z)*xy30) + ((*s)(3)*(*s)(1)*xy31) + ((*s)(3)*(*s)(2)*xy32) + ((*s)(3)*(*s)(3)*xy33);
    }

    else if ( isAltDiff() == 104 )
    {
        NiceAssert(s);

        Matrix<double> xxyy(4,4);

        xxyy("&",z,z) = xy00; xxyy("&",z,1) = xy10; xxyy("&",z,2) = xy20; xxyy("&",z,3) = xy30;
        xxyy("&",1,z) = xy10; xxyy("&",1,1) = xy11; xxyy("&",1,2) = xy21; xxyy("&",1,3) = xy31;
        xxyy("&",2,z) = xy20; xxyy("&",2,1) = xy21; xxyy("&",2,2) = xy22; xxyy("&",2,3) = xy32;
        xxyy("&",3,z) = xy30; xxyy("&",3,1) = xy31; xxyy("&",3,2) = xy32; xxyy("&",3,3) = xy33;

        res  =  xxyy((*s)(z),(*s)(z)) - xxyy((*s)(z),(*s)(1));
        res += -xxyy((*s)(1),(*s)(z)) + xxyy((*s)(1),(*s)(1));

        res +=  xxyy((*s)(2),(*s)(2)) - xxyy((*s)(2),(*s)(3));
        res += -xxyy((*s)(3),(*s)(2)) + xxyy((*s)(3),(*s)(3));
    }

    return;
}

inline
void MercerKernel::diffmnorm(int m, gentype &res, const gentype &xyprod, const Vector<const gentype *> &xanorm, const Matrix<double> &xxyy, const Vector<int> *ss) const
{
    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    int i,j;

    if ( isAltDiff() == 0 )
    {
        res  = xyprod;
        res *= -m;

        if ( m )
        {
            int i;

            for ( i = 0 ; i < m ; i++ )
            {
                res += (*(xanorm(i)));
            }
        }
    }

    else if ( isAltDiff() == 1 )
    {
        res  = xyprod;
        res *= -2.0;

        if ( m )
        {
            int i;

            for ( i = 0 ; i < m ; i++ )
            {
                res += (*(xanorm(i)));
            }
        }
    }

    else if ( isAltDiff() == 2 )
    {
        res = 0.0;

        if ( m )
        {
            for ( i = 0 ; i < m ; i++ )
            {
                res += (*(xanorm(i)));
            }

            for ( i = 0 ; i < m ; i++ )
            {
                for ( j = 0 ; j < m ; j++ )
                {
                    res -= (xxyy(i,j))/((double) m);
                }
            }
        }

        res *= 2.0;
    }

    else if ( isAltDiff() == 103 )
    {
        res = 0.0;

        NiceAssert(ss);

        for ( i = 0 ; i < m ; i++ )
        {
            for ( j = 0 ; j < m ; j++ )
            {
                res += (*ss)(i)*(*ss)(j)*xxyy(i,j);
            }
        }
    }

    else if ( isAltDiff() == 104 )
    {
        res = 0.0;

        NiceAssert(ss);

        for ( i = 0 ; i < m ; i += 2 )
        {
            res +=  xxyy((*ss)(i  ),(*ss)(i  )) - xxyy((*ss)(i  ),(*ss)(i+1));
            res += -xxyy((*ss)(i+1),(*ss)(i  )) + xxyy((*ss)(i+1),(*ss)(i+1));
        }
    }

    return;
}

inline
void MercerKernel::diff3norm(double &res, const double &xyprod, const double &xanorm, const double &xbnorm, const double &xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s) const
{
    (void) s;

    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    if ( isAltDiff() == 0 )
    {
        res = xanorm+xbnorm+xcnorm-(3*xyprod);
    }

    else if ( isAltDiff() == 1 )
    {
        res = xanorm+xbnorm+xcnorm-(2*xyprod);
    }

    else if ( isAltDiff() == 2 )
    {
        res = xanorm+xbnorm+xcnorm;

        res -= ( xy00 + xy10 + xy20 )/3.0;
        res -= ( xy10 + xy11 + xy21 )/3.0;
        res -= ( xy10 + xy21 + xy22 )/3.0;

        res *= 2.0;
    }

    else
    {
        throw("diff3norm not defined for altdiff != 0,1,2");
    }

    return;
}

inline
void MercerKernel::diff4norm(double &res, const double &xyprod, const double &xanorm, const double &xbnorm, const double &xcnorm, const double &xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const
{
    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    int z = 0;

    if ( isAltDiff() == 0 )
    {
        res = xanorm+xbnorm+xcnorm+xdnorm-(4*xyprod);
    }

    else if ( isAltDiff() == 1 )
    {
        res = xanorm+xbnorm+xcnorm+xdnorm-(2*xyprod);
    }

    else if ( isAltDiff() == 2 )
    {
        res = xanorm+xbnorm+xcnorm+xdnorm;

        res -= ( xy00 + xy10 + xy20 + xy30)/4.0;
        res -= ( xy10 + xy11 + xy21 + xy31)/4.0;
        res -= ( xy20 + xy21 + xy22 + xy32)/4.0;
        res -= ( xy30 + xy31 + xy32 + xy33)/4.0;

        res *= 2.0;
    }

    else if ( isAltDiff() == 103 )
    {
        NiceAssert(s);

        res  = ((*s)(z)*(*s)(z)*xy00) + ((*s)(z)*(*s)(1)*xy10) + ((*s)(z)*(*s)(2)*xy20) + ((*s)(z)*(*s)(3)*xy30);
        res += ((*s)(1)*(*s)(z)*xy10) + ((*s)(1)*(*s)(1)*xy11) + ((*s)(1)*(*s)(2)*xy21) + ((*s)(1)*(*s)(3)*xy31);
        res += ((*s)(2)*(*s)(z)*xy20) + ((*s)(2)*(*s)(1)*xy21) + ((*s)(2)*(*s)(2)*xy22) + ((*s)(2)*(*s)(3)*xy32);
        res += ((*s)(3)*(*s)(z)*xy30) + ((*s)(3)*(*s)(1)*xy31) + ((*s)(3)*(*s)(2)*xy32) + ((*s)(3)*(*s)(3)*xy33);
    }

    else if ( isAltDiff() == 104 )
    {
        NiceAssert(s);

        Matrix<double> xxyy(4,4);

        xxyy("&",z,z) = xy00; xxyy("&",z,1) = xy10; xxyy("&",z,2) = xy20; xxyy("&",z,3) = xy30;
        xxyy("&",1,z) = xy10; xxyy("&",1,1) = xy11; xxyy("&",1,2) = xy21; xxyy("&",1,3) = xy31;
        xxyy("&",2,z) = xy20; xxyy("&",2,1) = xy21; xxyy("&",2,2) = xy22; xxyy("&",2,3) = xy32;
        xxyy("&",3,z) = xy30; xxyy("&",3,1) = xy31; xxyy("&",3,2) = xy32; xxyy("&",3,3) = xy33;

        res  =  xxyy((*s)(z),(*s)(z)) - xxyy((*s)(z),(*s)(1));
        res += -xxyy((*s)(1),(*s)(z)) + xxyy((*s)(1),(*s)(1));

        res +=  xxyy((*s)(2),(*s)(2)) - xxyy((*s)(2),(*s)(3));
        res += -xxyy((*s)(3),(*s)(2)) + xxyy((*s)(3),(*s)(3));
    }

    return;
}

inline
void MercerKernel::diffmnorm(int m, double &res, const double &xyprod, const Vector<const double *> &xanorm, const Matrix<double> &xxyy, const Vector<int> *ss) const
{
    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    int i,j;

    if ( isAltDiff() == 0 )
    {
        res = -m*xyprod;

        if ( m )
        {
            int i;

            for ( i = 0 ; i < m ; i++ )
            {
                res += *(xanorm(i));
            }
        }
    }

    else if ( isAltDiff() == 1 )
    {
        res = -2*xyprod;

        if ( m )
        {
            int i;

            for ( i = 0 ; i < m ; i++ )
            {
                res += *(xanorm(i));
            }
        }
    }

    else if ( isAltDiff() == 2 )
    {
        res = 0.0;

        if ( m )
        {
            for ( i = 0 ; i < m ; i++ )
            {
                res += (*xanorm(i));
            }

            for ( i = 0 ; i < m ; i++ )
            {
                for ( j = 0 ; j < m ; j++ )
                {
                    res -= xxyy(i,j)/m;
                }
            }
        }

        res *= 2.0;
    }

    else if ( isAltDiff() == 103 )
    {
        res = 0.0;

        NiceAssert(ss);

        for ( i = 0 ; i < m ; i++ )
        {
            for ( j = 0 ; j < m ; j++ )
            {
                res += (*ss)(i)*(*ss)(j)*xxyy(i,j);
            }
        }
    }

    else if ( isAltDiff() == 104 )
    {
        res = 0.0;

        NiceAssert(ss);

        for ( i = 0 ; i < m ; i += 2 )
        {
            res +=  xxyy((*ss)(i  ),(*ss)(i  )) - xxyy((*ss)(i  ),(*ss)(i+1));
            res += -xxyy((*ss)(i+1),(*ss)(i  )) + xxyy((*ss)(i+1),(*ss)(i+1));
        }
    }

    return;
}





inline int UPNTVI(int i, int off);
inline int UPNTVI(int i, int off)
{
//    static unsigned int ind = 0;
//
//    return -(((ind++)%81)+10);
    return -((10*off)+i);
}



inline
double &MercerKernel::K0(double &res,
                         const double &bias, const gentype **pxyprod,
                         int xdim, int xconsist, int xresmode, int mlid, int assumreal) const
{
    return yyyK0(res,bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,0);
}

inline
double &MercerKernel::K1(double &res,
                         const SparseVector<gentype> &x, 
                         const vecInfo &xinfo, 
                         const double &bias,
                         const gentype **pxyprod,
                         int i, 
                         int xdim, int xconsist, int resmode, int mlid, 
                         const double *xy, int assumreal) const
{
    return yyyK1(res,x,xinfo,bias,pxyprod,i,xdim,xconsist,resmode,mlid,xy,assumreal,0);
}

inline
double &MercerKernel::K2(double &res,
                         const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                         const vecInfo &xinfo, const vecInfo &yinfo,
                         const double &bias,
                         const gentype **pxyprod,
                         int i, int j,
                         int xdim, int xconsist, int resmode, int mlid, 
                         const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    return yyyK2(res,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal,0);
}

inline
double &MercerKernel::K3(double &res,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                         const double &bias, const gentype **pxyprod,
                         int ia, int ib, int ic, 
                         int xdim, int xconsist, int xresmode, int mlid, 
                         const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal) const
{
    return yyyK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,0);
}

inline
double &MercerKernel::K4(double &res,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                         const double &bias, const gentype **pxyprod,
                         int ia, int ib, int ic, int id,
                         int xdim, int xconsist, int xresmode, int mlid, 
                         const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal) const
{
    return yyyK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,assumreal,0);
}


inline
double &MercerKernel::Km(int m,
                         double &res,
                         Vector<const SparseVector<gentype> *> &x,
                         Vector<const vecInfo *> &xinfo,
                         const double &bias,
                         Vector<int> &i,
                         const gentype **pxyprod,
                         int xdim, int xconsist, int resmode, int mlid, 
                         const Matrix<double> *xy, int assumreal) const
{
    return yyyKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,assumreal,0);
}

// ===================================================================================

inline
gentype &MercerKernel::K0(gentype &res,
                          const gentype &bias,
                          const gentype **pxyprod,
                          int xdim, int xconsist, int xresmode, int mlid, int assumreal) const
{
    return yyyK0(res,bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,0);
}

inline
gentype &MercerKernel::K1(gentype &res,
                          const SparseVector<gentype> &x, 
                          const vecInfo &xinfo, 
                          const gentype &bias,
                          const gentype **pxyprod,
                          int i, 
                          int xdim, int xconsist, int resmode, int mlid, 
                          const double *xy, int assumreal) const
{
    return yyyK1(res,x,xinfo,bias,pxyprod,i,xdim,xconsist,resmode,mlid,xy,assumreal,0);
}

inline
gentype &MercerKernel::K2(gentype &res,
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                          const vecInfo &xinfo, const vecInfo &yinfo,
                          const gentype &bias,
                          const gentype **pxyprod,
                          int i, int j,
                          int xdim, int xconsist, int resmode, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    return yyyK2(res,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal,0);
}

inline
gentype &MercerKernel::K3(gentype &res,
                          const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                          const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                          const gentype &bias,
                          const gentype **pxyprod,
                          int ia, int ib, int ic, 
                          int xdim, int xconsist, int xresmode, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal) const
{
    return yyyK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,0);
}

inline
gentype &MercerKernel::K4(gentype &res,
                          const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                          const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                          const gentype &bias,
                          const gentype **pxyprod,
                          int ia, int ib, int ic, int id, 
                          int xdim, int xconsist, int xresmode, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal) const
{
    return yyyK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,assumreal,0);
}

inline
gentype &MercerKernel::Km(int m, gentype &res,
                          Vector<const SparseVector<gentype> *> &x,
                          Vector<const vecInfo *> &xinfo,
                          const gentype &bias,
                          Vector<int> &i,
                          const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, 
                          const Matrix<double> *xy, int assumreal) const
{
    return yyyKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,assumreal,0);
}

// ====================================================================================

inline
double MercerKernel::distK(const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                           const vecInfo &xinfo, const vecInfo &yinfo, 
                           int i, int j, 
                           int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    double res = 0;
    double temp = 0;

    const gentype bias(0);

    if ( !isSimpleDistKernel() )
    {
        res += K2(temp,x,x,xinfo,xinfo,bias,NULL,i,i,xdim,xconsist,0,mlid,xy00,xy00,xy00,assumreal);
        res += K2(temp,y,y,yinfo,yinfo,bias,NULL,j,j,xdim,xconsist,0,mlid,xy11,xy11,xy11,assumreal);
    }

    res -= 2*K2(temp,x,y,xinfo,yinfo,bias,NULL,i,j,xdim,xconsist,0,mlid,xy00,xy10,xy11,assumreal);

    return res;
}

inline
void MercerKernel::ddistKdx(double &xscaleres, double &yscaleres, 
                            int &minmaxind, 
                            const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                            const vecInfo &xinfo, const vecInfo &yinfo, 
                            int i, int j, 
                            int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    gentype xres;
    gentype yres;
    gentype bres;
    gentype dummybias(0);

    dK2delx(xres,yres,minmaxind,x,y,xinfo,yinfo,dummybias,NULL,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);

    xscaleres = -2*((double) xres);
    yscaleres = -2*((double) yres);

    if ( !isSimpleDistKernel() )
    {
        dK2delx(xres,yres,minmaxind,x,x,xinfo,xinfo,dummybias,NULL,i,i,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);

        xscaleres += (double) xres;
        xscaleres += (double) yres;

        dK2delx(xres,yres,minmaxind,y,y,yinfo,yinfo,dummybias,NULL,j,j,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);

        yscaleres += (double) xres;
        yscaleres += (double) yres;
    }

    return;
}

//phantomx
inline
void MercerKernel::densedKdx(double &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double &bias, int i, int j, int xdim, int xconsist, int mlid, int assumreal) const
{
    NiceAssert( !isprod );

    // x now appropriately constructed as required

    gentype xyprod;
    gentype yxprod;

    if ( assumreal )
    {
        xyprod.force_double();
        yxprod.force_double();
    }

    twoProductDiverted(xyprod,x,y,xconsist,assumreal);
    twoProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

    xyprod += bias;
    yxprod += bias;

    // NB: second last argument 1 to indicate dense derivative
    gentype temp;
    K2i(temp,xyprod,yxprod,xinfo,yinfo,getmnorm(xinfo,x,2,xconsist,assumreal),getmnorm(yinfo,y,2,xconsist,assumreal),x,y,i,j,xdim,1,0,mlid,0,size(),assumreal);
    res = (double) temp;

    return;
}

//phantomx
inline
void MercerKernel::denseintK(double &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const double &bias, int i, int j, int xdim, int xconsist, int mlid, int assumreal) const
{
    NiceAssert( !isprod );

    // x now appropriately constructed as required

    gentype xyprod;
    gentype yxprod;

    if ( assumreal )
    {
        xyprod.force_double();
        yxprod.force_double();
    }

    twoProductDiverted(xyprod,x,y,xconsist,assumreal);
    twoProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

    xyprod += bias;
    yxprod += bias;

    // NB: second last argument x to indicate dense integral
    gentype temp;
    K2i(temp,xyprod,yxprod,xinfo,yinfo,getmnorm(xinfo,x,2,xconsist,assumreal),getmnorm(yinfo,y,2,xconsist,assumreal),x,y,i,j,xdim,2,0,mlid,0,size(),assumreal);
    res = (double) temp;

    return;
}

// ====================================================================================

inline
void MercerKernel::dK(double &xygrad, double &xnormgrad, int &minmaxind,
                        const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                        const vecInfo &xinfo, const vecInfo &yinfo,
                        const double &bias,
                        const gentype **pxyprod,
                        int i, int j,
                        int xdim, int xconsist, int mlid, 
                        const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    yyydKK2(xygrad,xnormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}

inline
void MercerKernel::dK(gentype &xygrad, gentype &xnormgrad, int &minmaxind,
                         const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                         const vecInfo &xinfo, const vecInfo &yinfo,
                         const gentype &bias,
                         const gentype **pxyprod,
                         int i, int j,
                         int xdim, int xconsist, int mlid, 
                         const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    yyydKK2(xygrad,xnormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}


// ===============================================================================================


inline
void MercerKernel::dK2delx(gentype &xscaleres, gentype &yscaleres,  int &minmaxind,
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          const gentype &bias, 
                          const gentype **pxyprod,
                          int i, int j, 
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    gentype xygrad(0.0);
    gentype xnormgrad(0.0);

    dK(xygrad,xnormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,1,assumreal);

    xscaleres  = xnormgrad;
    xscaleres *= 2.0;
    yscaleres  = xygrad;

    return;
}

inline
void MercerKernel::dK2delx(double &xscaleres, double &yscaleres,  int &minmaxind,
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          const double &bias,
                          const gentype **pxyprod, 
                          int i, int j,
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    double xygrad = 0.0;
    double xnormgrad = 0.0;

    dK(xygrad,xnormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,1,assumreal);

    xscaleres  = xnormgrad;
    xscaleres *= 2.0;
    yscaleres  = xygrad;

    return;
}

inline
void MercerKernel::dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, 
                          const Vector<int> &q, 
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          const gentype &bias, 
                          const gentype **pxyprod, 
                          int i, int j, 
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int deepDerive, int assumreal) const
{
    int z = 0;

    if ( q.size() == 0 )
    {
        // "no gradient" case

        sc.resize(1);
        n.resize(1);

        n("&",z).resize(z);

        K2(sc("&",z),x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,0,mlid,xy00,xy10,xy11,assumreal);
    }

    else if ( q.size() == 1 )
    {
        if ( q(z) == 0 )
        {
            // d/dx case - result is sc(0).x + sc(1).y

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);
            
            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            dK2delx(sc("&",z),sc("&",1),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);
        }

        else
        {
            // d/dy case - result is sc(0).x + sc(1).y
            // We assume symmetry to evaluate this

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);
            
            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            dK2delx(sc("&",1),sc("&",z),minmaxind,y,x,yinfo,xinfo,bias,NULL,j,i,xdim,xconsist,mlid,NULL,NULL,NULL,assumreal);
        }
    }

    else if ( q.size() == 2 )
    {
        if ( ( q(z) == 0 ) && ( q(1) == 0 ) )
        {
            // d^2/dx^2 case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdelx(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,assumreal);
        }

        else if ( ( q(z) == 0 ) && ( q(1) == 1 ) )
        {
            // d/dx d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdely(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,assumreal);
        }

        else if ( ( q(z) == 1 ) && ( q(1) == 0 ) )
        {
            // d/dy d/dx case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdely(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,NULL,j,i,xdim,xconsist,mlid,NULL,NULL,NULL,deepDerive,assumreal);
        }

        else
        {
            // d/dy d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdelx(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,NULL,j,i,xdim,xconsist,mlid,NULL,NULL,NULL,deepDerive,assumreal);
        }
    }

    else
    {
        yyydnKK2del(sc,n,minmaxind,q,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDerive);
    }

    return;
}

inline
void MercerKernel::dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, 
                          const Vector<int> &q, 
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          const double  &bias, 
                          const gentype **pxyprod, 
                          int i, int j, 
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int deepDerive, int assumreal) const
{
    int z = 0;

    if ( q.size() == 0 )
    {
        // "no gradient" case

        sc.resize(1);
        n.resize(1);

        n("&",z).resize(z);

        K2(sc("&",z),x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,0,mlid,xy00,xy10,xy11,assumreal);
    }

    else if ( q.size() == 1 )
    {
        if ( q(z) == 0 )
        {
            // d/dx case - result is sc(0).x + sc(1).y

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);
            
            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            dK2delx(sc("&",z),sc("&",1),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);
        }

        else
        {
            // d/dy case - result is sc(0).x + sc(1).y
            // We assume symmetry to evaluate this

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);
            
            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            dK2delx(sc("&",1),sc("&",z),minmaxind,y,x,yinfo,xinfo,bias,NULL,j,i,xdim,xconsist,mlid,NULL,NULL,NULL,assumreal);
        }
    }

    else if ( q.size() == 2 )
    {
        if ( ( q(z) == 0 ) && ( q(1) == 0 ) )
        {
            // d^2/dx^2 case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdelx(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,assumreal);
        }

        else if ( ( q(z) == 0 ) && ( q(1) == 1 ) )
        {
            // d/dx d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdely(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,assumreal);
        }

        else if ( ( q(z) == 1 ) && ( q(1) == 0 ) )
        {
            // d/dy d/dx case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdely(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,NULL,j,i,xdim,xconsist,mlid,NULL,NULL,NULL,deepDerive,assumreal);
        }

        else
        {
            // d/dy d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdelx(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,NULL,j,i,xdim,xconsist,mlid,NULL,NULL,NULL,deepDerive,assumreal);
        }
    }

    else
    {
        yyydnKK2del(sc,n,minmaxind,q,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDerive);
    }

    return;
}




inline
void MercerKernel::d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 const gentype &bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dx_idx_j = d2K/dada da/dx_i 2x_j + d2K/dzda dz/dx_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz da/dx_i y_j + d2K/dzdz dz/dx_i y_j
    //              = d2K/dada 2x_i 2x_j + d2K/dzda y_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz 2x_i y_j + d2K/dzdz y_i y_j
    //
    // d2K/dxdx = 4 d2K/dada x.x' + 2 d2K/dzda y.x' + 2 d2K/dadz x.y' + d2K/dzdz y.y' + 2 dK/da I

    gentype xygrad;
    gentype xnormgrad;
    gentype xyxygrad;
    gentype xyxnormgrad;
    gentype xyynormgrad;
    gentype xnormxnormgrad;
    gentype xnormynormgrad;
    gentype ynormynormgrad;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDeriv,assumreal);

    xxscaleres = 4.0*xnormxnormgrad;
    xyscaleres = 2.0*xyxnormgrad;
    yxscaleres = xyscaleres;
    yyscaleres = xyxygrad;
    constres   = 2.0*xnormgrad;

    return;
}

inline
void MercerKernel::d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 const gentype &bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dy_idx_j = d2K/dzda dz/dy_i 2x_j + d2K/dbda db/dy_i 2x_j + d2K/dzdz dz/dy_i y_j + d2K/dbdz db/dy_i y_j + dK/dz delta_{ij}
    //              = d2K/dzda x_i     2x_j + d2K/dbda 2y_i    2x_j + d2K/dzdz x_i     y_j + d2K/dbdz 2y_i    y_j + dK/dz delta_{ij}
    //              = 2 d2K/dzda x_i x_j + 4 d2K/dbda y_i x_j + d2K/dzdz x_i y_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dx_idy_j = 2 d2K/dzda x_i x_j + 4 d2K/dbda x_i y_j + d2K/dzdz y_i x_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dxdy = 2 d2K/dzda x.x' + 4 d2K/dbda x.y' + d2K/dzdz y.x' + 2 d2K/dbdz y.y' + dK/dz I

    gentype xygrad;
    gentype xnormgrad;
    gentype xyxygrad;
    gentype xyxnormgrad;
    gentype xyynormgrad;
    gentype xnormxnormgrad;
    gentype xnormynormgrad;
    gentype ynormynormgrad;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDeriv,assumreal);

    xxscaleres = 2.0*xyxnormgrad;
    xyscaleres = 4.0*xnormynormgrad;
    yxscaleres = xyxygrad;
    yyscaleres = 2.0*xyynormgrad;
    constres   = xygrad;

    return;
}

inline
void MercerKernel::d2K2delxdelx(double &xxscaleres, double &yyscaleres, double &xyscaleres, double &yxscaleres, double &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 const double &bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dx_idx_j = d2K/dada da/dx_i 2x_j + d2K/dzda dz/dx_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz da/dx_i y_j + d2K/dzdz dz/dx_i y_j
    //              = d2K/dada 2x_i 2x_j + d2K/dzda y_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz 2x_i y_j + d2K/dzdz y_i y_j
    //
    // d2K/dxdx = 4 d2K/dada x.x' + 2 d2K/dzda y.x' + 2 d2K/dadz x.y' + d2K/dzdz y.y' + 2 dK/da I

    double xygrad;
    double xnormgrad;
    double xyxygrad;
    double xyxnormgrad;
    double xyynormgrad;
    double xnormxnormgrad;
    double xnormynormgrad;
    double ynormynormgrad;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDeriv,assumreal);

    xxscaleres = 4.0*xnormxnormgrad;
    xyscaleres = 2.0*xyxnormgrad;
    yxscaleres = xyscaleres;
    yyscaleres = xyxygrad;
    constres   = 2.0*xnormgrad;

    return;
}

inline
void MercerKernel::d2K2delxdely(double &xxscaleres, double &yyscaleres, double &xyscaleres, double &yxscaleres, double &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 const double &bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dy_idx_j = d2K/dzda dz/dy_i 2x_j + d2K/dbda db/dy_i 2x_j + d2K/dzdz dz/dy_i y_j + d2K/dbdz db/dy_i y_j + dK/dz delta_{ij}
    //              = d2K/dzda x_i     2x_j + d2K/dbda 2y_i    2x_j + d2K/dzdz x_i     y_j + d2K/dbdz 2y_i    y_j + dK/dz delta_{ij}
    //              = 2 d2K/dzda x_i x_j + 4 d2K/dbda y_i x_j + d2K/dzdz x_i y_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dx_idy_j = 2 d2K/dzda x_i x_j + 4 d2K/dbda x_i y_j + d2K/dzdz y_i x_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dxdy = 2 d2K/dzda x.x' + 4 d2K/dbda x.y' + d2K/dzdz y.x' + 2 d2K/dbdz y.y' + dK/dz I

    double xygrad;
    double xnormgrad;
    double xyxygrad;
    double xyxnormgrad;
    double xyynormgrad;
    double xnormxnormgrad;
    double xnormynormgrad;
    double ynormynormgrad;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDeriv,assumreal);

    xxscaleres = 2.0*xyxnormgrad;
    xyscaleres = 4.0*xnormynormgrad;
    yxscaleres = xyxygrad;
    yyscaleres = 2.0*xyynormgrad;
    constres   = xygrad;

    return;
}


inline
void MercerKernel::d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, 
         const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
         const vecInfo &xinfo, const vecInfo &yinfo, 
         const gentype &bias, const gentype **pxyprod, 
         int i, int j, 
         int xdim, int xconsist, int mlid, 
         const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    yyyd2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}

inline
void MercerKernel::d2K(double &xygrad, double &xnormgrad, double &xyxygrad, double &xyxnormgrad, double &xyynormgrad, double &xnormxnormgrad, double &xnormynormgrad, double &ynormynormgrad, int &minmaxind, 
         const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
         const vecInfo &xinfo, const vecInfo &yinfo, 
         const double  &bias, const gentype **pxyprod, 
         int i, int j, 
         int xdim, int xconsist, int mlid, 
         const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    yyyd2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}




// ====================================================================================


// "Reversing" functions.
//
// For speed of operation it is sometimes helpful to retrieve either the
// inner product or distance from an evaluated kernel.  These functions
// let you do that
//
// isReversible: test if kernel is reversible.  Output is:
//     0: kernel cannot be reversed
//     1: kernel can be reversed to produce <x,y>+bias
//     2: kernel can be reversed to produce ||x-y||^2
//
// reverseK: reverse kernel as described by isReversible
//
// The result so produced can be fed back in via the pxyprod argument
// (appropriately set) to speed up calculation of results.  Use case
// could be quickly changing kernel parameters with minimal recalculation.
//
// As a general rule these only work with isSimpleFastKernel or
// isSimpleKernelChain, and then in limited cases.  For the chain case
// the result is the relevant (processed) output of the first layer.

//phantomx
inline
int MercerKernel::isReversible(void) const
{
    int res = 0;

    if ( ( ( size() == 1 ) && isSimpleKernel() && churnInner() ) || ( ( size() == 2 ) && isSimpleKernelChain() && churnInner() ) )
    {
        const Vector<int> &ic = dIntConstants(size()-1);

        if (   ( cType(size()-1) == 1  )                          ||
             ( ( cType(size()-1) == 2  ) && !(ic(0)%2) && ic(0) ) ||
               ( cType(size()-1) == 7  )                             )
        {
            res = 1;
        }

        else if ( ( cType(size()-1) == 3  ) ||
                  ( cType(size()-1) == 4  ) ||
                  ( cType(size()-1) == 5  ) ||
                  ( cType(size()-1) == 14 ) ||
                  ( cType(size()-1) == 15 ) ||
                  ( cType(size()-1) == 23 ) ||
                  ( cType(size()-1) == 38 )    )
        {
            res = 2;
        }
    }

    return res;
}

//KERNELSHERE
//phantomx
inline
gentype &MercerKernel::reverseK(gentype &res, const gentype &Kval) const
{
    if ( ( ( size() == 1 ) && isSimpleKernel() ) || ( ( size() == 2 ) && isSimpleKernelChain() ) )
    {
        res /= cWeight(size()-1);

        retVector<gentype> tmpva;

        const Vector<gentype> &r = dRealConstants(size()-1)(1,1,dRealConstants(size()-1).size()-1,tmpva);
        const Vector<int> &ic = dIntConstants(size()-1);

        if ( cType(size()-1) == 1 )
        {
            res  = Kval;
            res *= r(0);
            res *= r(0);
        }

        else if ( ( cType(size()-1) == 2  ) && !(ic(0)%2) && ic(0) )
        {
            res  = Kval;
            raiseto(res,oneintgentype()/((double) ic(0)));
            res -= r(1);
            res *= r(0);
            res *= r(0);
        }

        else if ( cType(size()-1) == 7 )
        {
            res  = Kval;
            OP_atanh(res);
            res -= r(1);
            res *= r(0);
            res *= r(0);
        }


        else if ( cType(size()-1) == 3  )
        {
            res  = Kval;
            OP_log(res);
            res += r(1);
            res.negate();
            res *= r(0);
            res *= r(0);
            res *= 2;
        }

        else if ( cType(size()-1) == 4  )
        {
            res  = Kval;
            OP_log(res);
            res += r(1);
            res.negate();
            res *= r(0);
            raiseto(res,2);
        }

        else if ( cType(size()-1) == 5  )
        {
            res  = Kval;
            OP_log(res);
            res += r(2);
            res.negate();
            res *= r(1);
            res *= pow(r(0),r(1));
            raiseto(res,twointgentype()/r(1));
        }

        else if ( cType(size()-1) == 14 )
        {
            res  = Kval;
            res.negate();
            raiseto(res,twointgentype()/r(1));
            res *= r(0);
            res *= r(0);
        }

        else if ( cType(size()-1) == 15 )
        {
            res  = Kval;
            res.negate();
            OP_exp(res);
            res -= 1;
            raiseto(res,twointgentype()/r(1));
            res *= r(0);
            res *= r(0);
        }

        else if ( cType(size()-1) == 23 )
        {
            res  = Kval;
            res.inverse();
            res -= 1;
            res *= r(0);
            res *= r(0);
        }

        else if ( cType(size()-1) == 38 )
        {
            res  = Kval;
            OP_log(res);
            res.negate();
            res *= r(0);
            raiseto(res,2);
        }

        else if ( cType(size()-1) == 41 )
        {
            res  = Kval;
            raiseto(res,twointgentype()*r(0)*r(0));
        }

        else
        {
            throw("phooey");
        }

        // Secondary scaling

        if ( isSimpleKernelChain() )
        {
            res /= cWeight(0);
        }
    }

    else
    {
        throw("wassseirefwn");
    }

    return res;
}

//KERNELSHERE
//phantomx
inline
double  &MercerKernel::reverseK(double &res, const double &Kval) const
{
    if ( ( ( size() == 1 ) && isSimpleKernel() ) || ( ( size() == 2 ) && isSimpleKernelChain() ) )
    {
        res /= (double) cWeight(size()-1);

        retVector<gentype> tmpva;

        const Vector<gentype> &r = dRealConstants(size()-1)(1,1,dRealConstants(size()-1).size()-1,tmpva);
        const Vector<int> &ic = dIntConstants(size()-1);

        if ( cType(size()-1) == 1  )
        {
            res =  Kval;
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( ( cType(size()-1) == 2  ) && !(ic(0)%2) && ic(0) )
        {
            res  = pow(Kval,1/((double) ic(0)));
            res -= (double) r(1);
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( cType(size()-1) == 7  )
        {
            res  = atanh(Kval);
            res -= (double) r(1);
            res /= (double) r(0);
        }


        else if ( cType(size()-1) == 3  )
        {
            res  = -log(Kval);
            res -= (double) r(1);
            res *= (double) r(0);
            res *= (double) r(0);
            res *= 2;
        }

        else if ( cType(size()-1) == 4  )
        {
            res  = -log(Kval);
            res -= (double) r(1);
            res *= (double) r(0);
            res *= res;
        }

        else if ( cType(size()-1) == 5  )
        {
            res  = -log(Kval);
            res -= (double) r(2);
            res *= (double) r(1);
            res *= (double) pow((double) r(0),(double) r(1));
            res  = pow(res,2/((double) r(1)));
        }

        else if ( cType(size()-1) == 14 )
        {
            res  = -Kval;
            res  = pow(res,2/((double) r(1)));
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( cType(size()-1) == 15 )
        {
            res  = -Kval;
            res  = exp(res);
            res -= 1;
            res  = pow(res,2/((double) r(1)));
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( cType(size()-1) == 23 )
        {
            res  = 1/Kval;
            res -= 1;
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( cType(size()-1) == 38 )
        {
            res  = -log(Kval);
            res *= (double) r(0);
            res *= res;
        }

        else if ( cType(size()-1) == 41 )
        {
            res  = pow(Kval,2*((double) r(0))*((double) r(0)));
        }

        else
        {
            throw("cor blimey!");
        }

        // Secondary scaling

        if ( isSimpleKernelChain() )
        {
            res /= (double) cWeight(0);
        }
    }

    else
    {
        gentype tempres(res);
        gentype tempKval(Kval);

        reverseK(tempres,tempKval);

        res = (double) tempres;
    }

    return res;
}





// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================




























// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================
// ===================================================================================


inline
double &MercerKernel::K0ip(double &res, 
                         const double &bias, const gentype **pxyprod,
                         int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK0(res,bias,pxyprod,xdim,xconsist,0,mlid,assumreal,1);
}

inline
double &MercerKernel::K1ip(double &res,
                        const SparseVector<gentype> &x,
                        const vecInfo &xinfo,
                        const double &bias, const gentype **pxyprod,
                        int i,
                        int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK1(res,x,xinfo,bias,pxyprod,i,xdim,xconsist,0,mlid,NULL,assumreal,1);
}

inline
double &MercerKernel::K2ip(double &res,
                        const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                        const vecInfo &xinfo, const vecInfo &yinfo,
                        const double &bias, const gentype **pxyprod,
                        int i, int j,
                        int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK2(res,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,0,mlid,NULL,NULL,NULL,assumreal,1);
}

inline
double &MercerKernel::K3ip(double &res,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                         const double &bias, const gentype **pxyprod,
                         int ia, int ib, int ic,
                         int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,0,mlid,NULL,NULL,NULL,NULL,NULL,NULL,assumreal,1);
}

inline
double &MercerKernel::K4ip(double &res,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                         const double &bias, const gentype **pxyprod,
                         int ia, int ib, int ic, int id,
                         int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,0,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,assumreal,1);
}

inline
double &MercerKernel::Kmip(int m,
                         double &res,
                         Vector<const SparseVector<gentype> *> &x,
                         Vector<const vecInfo *> &xinfo,
                         Vector<int> &i,
                         const double &bias, const gentype **pxyprod,
                         int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,0,mlid,NULL,assumreal,1);
}












// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================


//FIXME: define dKKprodip, dKKprodd for inner product and diffis derivatives
//FIXME: then make dKdz, dKdz functions that follow same form as K to this level, then pass off into base afterwards (with throw if m > 2 and isaltdiff != 0 for simplicity)

template <class T>
int MercerKernel::KKpro(T &totres, const T &inxyprod, const T &diffis, int *i, int locindstart, int locindend, int xdim, int m, T &logres, const T *xprod) const
{
    NiceAssert( ( m == 2 ) || ( ismagterm == zeroint() ) );

    T xyprod = inxyprod;

    totres = xyprod;

    int logresvalid = 0;

    if ( !isFastKernelSum() && ( locindstart > locindend ) )
    {
        return 0;
    }

    else if ( locindstart == locindend )
    {
        // shortcut version.

        int ind = locindstart;
        {
            logresvalid = 0;

            T &res = totres;

            retVector<gentype> tmpva;

            const Vector<gentype> &r = dRealConstants(ind)(1,1,dRealConstants(ind).size()-1,tmpva);
            const Vector<int> &ic = dIntConstants(ind);

            int ktype = cType(ind);

            KKprosingle(res,xyprod,diffis,i,xdim,m,logres,xprod,ktype,logresvalid,cWeight(ind),r,ic,isMagTerm(ind));
        }
    }

    else
    {
        retVector<int>     tmpva;
        retVector<gentype> tmpvb;

        (void) tmpva;

        NiceAssert( isFastKernelSum() || ( ismagterm(locindstart,1,locindend,tmpva) == zeroint() ) );

        int ind = locindstart;

        // NB: xyprod is used at all layers for chained kernels
        //     diffis is used only at first layer for chained kernels

        Vector<T> allres(locindend-locindstart+1);

        Vector<T> allxygrad(locindend-locindstart+1);
        Vector<T> alldiffgrad(locindend-locindstart+1);

        for ( ; ind <= locindend ; ind++ )
        {
            logresvalid = 0; // only final in chain can set logres

            T &res = allres("&",ind-locindstart);

            const Vector<gentype> &r = dRealConstants(ind)(1,1,dRealConstants(ind).size()-1,tmpvb);
            const Vector<int> &ic = dIntConstants(ind);

            int ktype = cType(ind);

            KKprosingle(res,xyprod,diffis,i,xdim,m,logres,xprod,ktype,logresvalid,cWeight(ind),r,ic,isMagTerm(ind));

            if ( isFastKernelSum() && ( ind == locindstart ) )
            {
                totres = res;
            }

            else if ( isFastKernelSum() )
            {
                totres += res;
            }

            else if ( ( ind == locindend ) && ( ind == locindstart ) )
            {
                totres = res;
            }

            else if ( ind == locindstart )
            {
                xyprod = res;
            }

            else if ( ind < locindend )
            {
                xyprod = res;
            }

            else
            {
                totres = res;
            }

            //if ( isSplit(ind) )
            //{
            //    ind++;
            //    break;
            //}
        }
    }

    return logresvalid && ( !isFastKernelSum() || ( locindstart == locindend ) );
}

//KERNELSHERE
//phantomx

template <class T>
int MercerKernel::KKprosingle(T &res, const T &xyprod, const T &diffis, int *i, int xdim, int m, T &logres, const T *xprod, int ktype, int &logresvalid, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const
{
    if ( magterm )
    {
        NiceAssert( m == 2 );

        res         = (const T &) weight;
        logres      = log((const T &) weight);
        logresvalid = 0;

        int retval = 0;
        int j,k;

        T altdiffis; altdiffis = 0.0;
        gentype altweight(1.0);

        int *ix; MEMNEWARRAY(ix,int,m);
        T *xxprod; MEMNEWARRAY(xxprod,T,m);

        for ( j = 0 ; j < m ; j++ )
        {
            T resx;    resx    = 0.0;
            T logresx; logresx = 0.0;
            int logresxvalid = 0;

            for ( k = 0 ; k < m ; k++ )
            {
                ix[k]     = i[j];
                xxprod[k] = xprod[j];
            }

            int retvalx = KKprosingle(resx,xprod[j],altdiffis,ix,xdim,m,logresx,xxprod,ktype,logresxvalid,altweight,r,ic,0);

            res         *= resx;
            logres      += logresx;
            logresvalid *= logresxvalid;

            retval += retvalx;
        }

        MEMDELARRAY(ix);
        MEMDELARRAY(xxprod);

        return retval;
    }

    res = 0.0;

    int retval = 0;

    switch ( ktype )
    {
        case 0:
        {
            // K = r1

            res = r(1);

            break;
        }

        case 1:
        {
            // K = z/(r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            break;
        }

        case 2:
        {
            // K = ( r1 + z/(r0.r0) )^i0

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1));
            raiseto(res,ic(0));

            break;
        }

        case 3:
        {
            // K = exp(-d/(2.r0.r0)-r1)

            res  = diffis;
            res *= -0.5;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scalsub(res,r(1));
            res += log(AltDiffNormConst(xdim,m,r(0)));

            logres      = res;
            logresvalid = 1;

            OP_exp(res);

            break;
        }

        case 4:
        {
            // K = exp(-sqrt(d)/r0)
            //
            // At d=0: dK = 0, d2K = 0

            T tmp(diffis);

            OP_sqrt(tmp);

            res = tmp;
            setnegate(res);
            scaldiv(res,r(0));
            scalsub(res,r(1));

            logres      = res;
            logresvalid = 1;

            OP_exp(res);

            break;
        }

        case 5:
        {
            // K = exp(-sqrt(d)^r1/(r1*r0^r1))
            //
            // At d=0, if r1 < 2: dK = 0, d2K = 0

            T tmpb = pow(sqrt(diffis),(T) r(1));
            T tmpc = pow((T) r(0),(T) r(1));

            res  = tmpb;
            res /= tmpc;
            scaldiv(res,r(1));
            setnegate(res);
            scalsub(res,r(2));

            logres      = res;
            logresvalid = 1;

            OP_exp(res);

            break;
        }

        case 7:
        {
            // K = tanh( z/(r0.r0) + r1 )

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1));
            OP_tanh(res);

            break;
        }

        case 8:
        {
            // K = ( 1 + d/(2*r0*r0*r1) )^(-r1)

            //OLD     8  | Rational quadratic     | 1 - d/(d+r0)
            //OLD     8  | Rational quadratic     | -1/(d+r0) + d/(d+r0)^2 = -K(x,y)/(d+r0)
            //OLD     8  | Rational quadratic     | K(x,y)/((d+r0)^2) - K(x,y)/((d+r0)^2) = 0

            T tmp;

            tmp  = diffis;
            tmp *= 0.5;
            scaldiv(tmp,r(0));
            scaldiv(tmp,r(0));
            scaldiv(tmp,r(1));
            tmp += 1.0;

            res = pow(res,-((T) r(1)));

//OLD            res = diffis;
//OLD            scaldiv(res,(diffis+r(0)));
//OLD            setnegate(res);
//OLD            res += 1.0;
//OLD
//OLD            xygrad = 0.0;
//OLD
//OLD            diffgrad = res;
//OLD            setnegate(diffgrad);
//OLD            scaldiv(res,(diffis+r(0)));

            break;
        }

        case 9:
        {
            // K = sqrt( d/(r0.r0) + r1^2 )
            //
            // if d/(r0.r0) + r1^2 = 0 then dk,d2K = 0

            res = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1),r(1));
            OP_sqrt(res);

            break;
        }

        case 10:
        {
            // K = 1/sqrt( d/(r0.r0) + r1^2 )
            //
            // Ill-defined as d + r1^2 -> 0

            res = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1),r(1));
            OP_sqrt(res);
            OP_einv(res);

            break;
        }

        case 11:
        {
            // K = 2/pi * arccos(-sqrt(d)/r0) - 2/pi * sqrt(d)/r0 * sqrt(1 - d/r0^2)

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));
            tempres *= -1.0;

            T tempsq(tempres);
            tempsq *= tempsq;
            tempsq *= -1.0;
            tempsq += 1.0;
            OP_sqrt(tempsq);

            if ( (double) abs2(tempres) < 1.0 )
            {
                // K = 2/pi * arccos(-sqrt(d)/r0) - 2/pi * sqrt(d)/r0 * sqrt(1 - d/r0^2)

                res  = tempsq;
                res *= tempres;
                res += acos(tempres);
                res *= NUMBASE_2ONPI;
            }

            break;
        }

        case 12:
        {
            // K = 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));

            // K = 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3

            res = tempres;
            res *= tempres;
            res *= tempres;
            res /= 3.0;
            res -= tempres;
            res /= 2.0;
            res *= 3.0;
            res += 1.0;

            //res = tempres;
            //res *= tempres;
            //res *= tempres;
            //res /= 2.0;
            //res *= 0.6666666666666666666666;
            //res -= tempres;
            //res /= 0.6666666666666666666666;
            //res += 1.0;

            break;
        }

        case 13:
        {
            // K = sinc(sqrt(d)/r0)
            //
            // if d = 0 then dK,d2K = 0

            T tmp(diffis);

            OP_sqrt(tmp);

            T tmpb(tmp);

            scaldiv(tmpb,r(0));

            res = tmpb;
            OP_sinc(res);

            break;
        }

        case 14:
        {
            // K = -sqrt(d)^r1
            //
            // if d = 0 then dK,d2K = 0

            res  = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            OP_sqrt(res);
            res  = pow(res,(T) r(1));
            res *= -1.0;

            break;
        }

        case 15:
        {
            // K = -log(sqrt(d)^r1 + 1)

            T tmpa(diffis);

            scaldiv(tmpa,r(0));
            scaldiv(tmpa,r(0));

            OP_sqrt(tmpa);

            T tmpb;

            tmpb = pow(tmpa,(T) r(1));
            tmpb += 1.0;

            res = tmpb;
            OP_log(res);
            setnegate(res);

            break;
        }

        case 19:
        {
            // K = 1/(1+(d/(r0.r0)))

            res = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            res += 1.0;
            OP_einv(res);

            break;
        }

        case 23:
        {
            // K = 1/(1+(sqrt(d)/r0)^r1)

            T tmp(diffis);

            OP_sqrt(tmp);
            scaldiv(tmp,r(0));

            res  = pow(tmp,(T) r(1));
            res += 1.0;
            OP_einv(res);

            break;
        }

        case 24:
        {
            // K = (1-((z/(r0.r0))^i0))/(1-(z/(r0.r0)))
            //
            // Ill-defined at z = 1

            T zsc(xyprod);

            scaldiv(zsc,r(0));
            scaldiv(zsc,r(0));

            if ( ( (double) abs2(zsc) > 1e-12 ) || ( ic(0) == 0 ) )
            {
                T tmp = zsc;

                raiseto(tmp,ic(0));
    
                res  = 1.0;
                res -= tmp;
                res /= (1.0-zsc);
            }

            break;
        }

        case 25:
        {
            // K = pi.cosh(pi-(sqrt(d)/r0))
    
            T tmp(diffis);

            OP_sqrt(tmp);

            res  = tmp;
            scaldiv(res,r(0));
            res -= NUMBASE_PI;
            res *= -1.0;
            OP_cosh(res);
            res *= NUMBASE_PI;

            break;
        }

        case 26:
        {
            // K = ((d/r0)^(r1+0.5))

            res  = diffis;
            scaldiv(res,r(0));
            res  = pow(res,((T) r(1))+0.5);

            break;
        }

        case 27:
        {
            // K = ((d/r0)^r1).ln(sqrt(d/r0))

            T scalres(diffis);
    
            scaldiv(scalres,r(0));
            OP_sqrt(scalres);

            T tempres(scalres);

            OP_log(tempres);
    
            scalres = pow(scalres,(T) r(1));

            res  = scalres;
            res *= tempres;

            break;
        }

        case 32:
        {
            // K = r1 if i == j >= 0, 0 otherwise

            if ( m )
            {
                if ( i[0] >= 0 )
                {
                    int j;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        if ( i[0] != i[j] )
                        {
                            break;
                        }
                    }

                    if ( j == m )
                    {
                        res = r(1);
                    }
                }
            }

            break;
        }

        case 33:
        {
            // K = 1/(2.r0) ( 1 if real(sqrt(d)) < r0, 0 otherwise )

            if ( real(sqrt(diffis)-r(0)) < zerogentype() )
            {
                res = 0.5/r(0);
            }

            break;
        }

        case 34:
        {
            // K = (1-sqrt(d)/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));

            if ( (double) real(tempres-1.0) < 0.0 )
            {
                res  = 1.0;
                res -= tempres;
                scaldiv(res,r(0));
            }

            break;
        }

        case 38:
        {
            // K = exp(-sqrt(d)/r0)

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));
            tempres *= -1.0;

            res = tempres;

            logres      = res;
            logresvalid = 1;

            OP_exp(res);

            break;
        }

        case 39:
        {
            // K = (1+((sqrt(3)/r0).sqrt(d))) . exp(-(sqrt(3)/r0).sqrt(d))

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));
            tempres *= sqrt(3.0);

            T expres(tempres);
            expres *= -1.0;
            OP_exp(expres);

            res  = tempres;
            res += 1;
            res *= expres;

            break;
        }
    
        case 42:
        {
            // K = agd(z/(r0.r0))

            T scalres = diffis;
            scaldiv(scalres,r(0));
            scaldiv(scalres,r(0));

            T scz = scalres;
            OP_sec(scz);
            scz *= scz;

            T taz = scalres;
            OP_tan(taz);

            res = scalres;
            OP_agd(res);

            break;
        }

        case 43:
        {
            // K = log((1+z/(r0.r0))/(1-z/(r0.r0)))

            T tempa = xyprod;

            tempa  = xyprod;
            scaldiv(tempa,r(0));
            scaldiv(tempa,r(0));

            T tempb = tempa;

            tempa += 1.0;
            tempb -= 1.0;

            res  = tempa;
            res /= tempb;
            OP_log(res);

            break;
        }

        case 44:
        {
            // K = exp(z/(r0.r0)-r1)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scalsub(res,r(1));

            logres      = res;
            logresvalid = 1;

            OP_exp(res);

            break;
        }

        case 45:
        {
            // K = sinh(z/(r0.r0))

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            OP_sinh(res);

            break;
        }

        case 46:
        {
            // K = cosh(z/(r0.r0))

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            OP_cosh(res);

            break;
        }

        case 47:
        {
            // K = sinc(sqrt(d)/r0).cos(2*pi*sqrt(d)/(r0.r1))
            //
            // if d = 0 then dK,d2K = 0

            T tmp(diffis);

            OP_sqrt(tmp);

            T tmpb(tmp);

            scaldiv(tmpb,r(0));

            res = tmpb;
            OP_sinc(res);

            T tmpc;

            tmpc = tmpb;
            tmpc *= (2*NUMBASE_PI);
            scaldiv(tmpc,r(1));
            OP_cos(tmpc);

            res *= tmpc;

            break;
        }

        case 100:
        {
            // K = z/(r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            break;
        }

        case 103:
        {
            // K = 0 if real(z) < 0, 1 otherwise

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = 0.0;
            }

            else
            {
                res = 1.0;
            }

            break;
        }

        case 104:
        {
            // K = 0 if real(z) < 0, z/(r0*r0) otherwise

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = 0.0;
            }

            else
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
            }

            break;
        }

        case 106:
        {
            // K = r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise

            static T zgt(0.0);

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            if ( xyprod < zgt )
            {
                scalmul(res,r(1));
            }

            break;
        }

        case 200:
        {
            // K = z/(r0.r0) - 1

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            res -= 1.0;

            break;
        }

        case 203:
        {
            // K = -1 if real(z) < 0, 1 otherwise

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = -1.0;
            }

            else
            {
                res = 1.0;
            }

            break;
        }

        case 204:
        {
            // K = 0 if real(z) < 0, z/(r0*r0) otherwise    - 1

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = -1.0;
            }

            else
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
                res -= 1.0;
            }

            break;
        }

        case 206:
        {
            // K = r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise

            static T zgt(0.0);

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            if ( xyprod < zgt )
            {
                scalmul(res,r(1));
            }

            res -= 1.0;

            break;
        }

        default:
        {
            throw("fee fi fo fum");

            retval = 1;
            break;
        }
    }

    res    *= (const T &) weight;
    logres += log((const T &) weight);

    return retval;
}


template <class T>
void MercerKernel::dKKpro(T &totxygrad, T &totxnormgrad, T &totres, const T &inxyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, const T &xxprod, const T &yyprod) const
{
    // Assumptions enforced here
    //
    // - magterm only included if this is a kernel sum
    // - if kernel chain then only the final term can use diffis
    // - no mixed sum/chain

    T xyprod = inxyprod;

    totres = xyprod;

    totxygrad    = 1.0;
    totxnormgrad = 0.0;

    if ( !isFastKernelSum() && ( locindstart > locindend ) )
    {
        return;
    }

    // Chained derivative is the prod: dkn(k{n-1}(...k1(k0(q)))) * dk{n-1}(...k1(k0(q)))... * dk1(k0(q)) * dk0(q)

    T totdiffgrad;

    totdiffgrad = 0.0;

    int ind;

    // NB: xyprod is used at all layers for chained kernels
    //     diffis is used only at first layer for chained kernels

    Vector<T> allres(locindend-locindstart+1);

    Vector<T> allxygrad(locindend-locindstart+1);
    Vector<T> alldiffgrad(locindend-locindstart+1);
    Vector<T> allxnormonlygrad(locindend-locindstart+1);

    for ( ind = locindstart ; ind <= locindend ; ind++ )
    {
        T &res = allres("&",ind-locindstart);

        T &xygrad        = allxygrad("&",ind-locindstart);
        T &diffgrad      = alldiffgrad("&",ind-locindstart);
        T &xnormonlygrad = allxnormonlygrad("&",ind-locindstart);

        retVector<gentype> tmpva;

        const Vector<gentype> &r = dRealConstants(ind)(1,1,dRealConstants(ind).size()-1,tmpva);
        const Vector<int> &ic = dIntConstants(ind);

        int ktype = cType(ind);

        dKKprosingle(xygrad,diffgrad,xnormonlygrad,res,xyprod,diffis,i,j,xdim,m,xxprod,yyprod,ktype,cWeight(ind),r,ic,isMagTerm(ind));

        if ( isFastKernelSum() && ( ind == locindstart ) )
        {
            totres = res;

            totxygrad    = xygrad;
            totdiffgrad  = diffgrad;
            totxnormgrad = xnormonlygrad;
        }

        else if ( isFastKernelSum() )
        {
            totres += res;

            totxygrad    += xygrad;
            totdiffgrad  += diffgrad;
            totxnormgrad += xnormonlygrad;
        }

        else if ( ( ind == locindend ) && ( ind == locindstart ) )
        {
            totres = res;

            totxygrad    = xygrad;
            totdiffgrad  = diffgrad;
            totxnormgrad = xnormonlygrad;
        }

        else if ( ind == locindstart )
        {
            xyprod = res;

            totxygrad    = xygrad;
            totdiffgrad  = diffgrad;
            totxnormgrad = xnormonlygrad;
        }

        else if ( ind < locindend )
        {
            xyprod = res;

            totxygrad    *= xygrad;
            totdiffgrad  *= xygrad; // diffgrad zero here by definition
            totxnormgrad *= xnormonlygrad; // xnormonlygrad likely
        }

        else
        {
            totres = res;

            totxygrad    *= xygrad;
            totdiffgrad  *= xygrad; // diffgrad zero here by definition
            totxnormgrad *= xnormonlygrad; // xnormonlygrad likely
        }
    }

    // diffis = ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    //
    // dK/dxyprod = partialK/partialxyprod + partialK/partialdiffis partialdiffis/partialxyprod
    //            = partialK/partialxyprod - 2*partialK/partialdiffis
    // dK/dxnorm  = partialK/partialdiffis partialdiffis/partialxnorm
    //            = partialK/partialdiffis 

    totxygrad -= totdiffgrad;
    totxygrad -= totdiffgrad;

    totxnormgrad += totdiffgrad;

    return;
}

//KERNELSHERE
//phantomx

template <class T>
void MercerKernel::dKKprosingle(T &xygrad, T &diffgrad, T &xnormonlygrad, T &res, const T &xyprod, const T &diffis, int i, int j, int xdim, int m, const T &xxprod, const T &yyprod, int ktype, const gentype &weight, const Vector<gentype> &r, const Vector<int> &ic, int magterm) const
{
    res = 0.0;

    xygrad        = 0.0;
    diffgrad      = 0.0;
    xnormonlygrad = 0.0;

    if ( magterm )
    {
        xygrad = 0.0;

        T xres; xres = 0.0;
        T yres; yres = 0.0;

        T ddgrad;         ddgrad         = 0.0;
        T ddiffgrad;      ddiffgrad      = 0.0;
        T ddnormonlygrad; ddnormonlygrad = 0.0;

        T altdiffis; altdiffis = 0.0;

        gentype altweight(1.0);

        T xxgrad; xxgrad = 0.0;

        dKKprosingle(xxgrad,ddiffgrad,ddnormonlygrad,xres,xxprod,altdiffis,i,i,xdim,m,xxprod,xxprod,ktype,altweight,r,ic,0);
        dKKprosingle(ddgrad,ddiffgrad,ddnormonlygrad,yres,yyprod,altdiffis,j,j,xdim,m,yyprod,yyprod,ktype,altweight,r,ic,0);

        (void) ddgrad;
        (void) ddiffgrad;
        (void) ddnormonlygrad;

        res  = xres;
        res *= yres;

        //xygrad  = 0.0;
        //xygrad *= yres;
        
        //diffgrad  = 0.0;
        //diffgrad *= yres;

        xnormonlygrad  = xxgrad;
        xnormonlygrad *= yres;

        res           *= (const T &) weight;
        //xygrad        *= (const T &) weight;
        //diffgrad      *= (const T &) weight;
        xnormonlygrad *= (const T &) weight;

        return;
    }

    switch ( ktype )
    {
        case 0:
        {
            // K = r1
            // dK = 0

            res = r(1);

            break;
        }

        case 1:
        {
            // K = z/(r0.r0)
            // dK = 1/(r0.r0)

            res    = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 2:
        {
            // K = ( r1 + z/(r0.r0) )^i0
            // dK = i0/(r0.r0) * ( r1 + z/(r0.r0) )^(i0-1)            if ( i0 >= 1 )

            T temp(xyprod);

            scaldiv(temp,r(0));
            scaldiv(temp,r(0));
            scaladd(temp,r(1));

            res = temp;
            raiseto(res,ic(0));

            if ( ic(0) >= 1 )
            {
                xygrad = temp;
                raiseto(xygrad,ic(0)-1);
                xygrad *= ic(0);
                scaldiv(xygrad,r(0));
                scaldiv(xygrad,r(0));
            }

            break;
        }

        case 3:
        {
            // K = exp(-d/(2.r0.r0))
            // dK = -K/(2*r0*r0)

            res  = diffis;
            res *= -0.5;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            res += log(AltDiffNormConst(xdim,m,r(0)));
            scalsub(res,r(1));
            OP_exp(res);

            diffgrad  = res;
            diffgrad *= -0.5;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            break;
        }

        case 4:
        {
            // K = exp(-sqrt(d)/r0)
            // dK = -K/(2*r0*sqrt(d))
            //
            // At d=0: dK = 0, d2K = 0

            T tmp(diffis);

            OP_sqrt(tmp);

            res = tmp;
            setnegate(res);
            scaldiv(res,r(0));
            scalsub(res,r(1));
            OP_exp(res);

            if ( (double) abs2(diffis) > 1e-12 )
            {
                diffgrad = res;
                diffgrad *= -0.5;
                scaldiv(diffgrad,r(0));
                diffgrad /= tmp;
            }

            break;
        }

        case 5:
        {
            // K = exp(-sqrt(d)^r1/(r1*r0^r1))
            // dK = -K*((sqrt(d)^(r1-2))/(2*r0^r1))
            //    = -K*((sqrt(d)^r1)/(2*d*r0^r1))
            //
            // At d=0, if r1 < 2: dK = 0, d2K = 0

            T tmpb = pow(sqrt(diffis),(T) r(1));
            T tmpc = pow((T) r(0),(T) r(1));

            res  = tmpb;
            res /= tmpc;
            scaldiv(res,r(1));
            setnegate(res);
            scalsub(res,r(2));
            OP_exp(res);

            if ( ( (double) abs2(diffis) > 1e-12 ) || ( (double) abs2(r(1)) >= 2.0 ) )
            {
                // dK = -K*((sqrt(d)^r1)/(2*d*r0^r1))

                diffgrad  = res;
                diffgrad *= tmpb;
                diffgrad *= -0.5;
                diffgrad /= diffis;
                diffgrad /= tmpc;
            }

            break;
        }

        case 7:
        {
            // K = tanh( z/(r0.r0) + r1 )
            // dK = 1/(r0.r0) * sech^2( r0 z + r1 )

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1));
            OP_tanh(res);

            xygrad = xyprod;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            scaladd(xygrad,r(1));
            OP_sech(xygrad);
            xygrad *= xygrad;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 8:
        {
            // K = ( 1 + d/(2*r0*r0*r1) )^(-r1)
            // dK = -(( 1 + d/(2*r0*r0*r1) )^(-r1-1))/(2*r0*r0)
            //    = -K/(2*r0*r0*( 1 + d/(2*r0*r0*r1) ))

            //OLD     8  | Rational quadratic     | 1 - d/(d+r0)
            //OLD     8  | Rational quadratic     | -1/(d+r0) + d/(d+r0)^2 = -K(x,y)/(d+r0)
            //OLD     8  | Rational quadratic     | K(x,y)/((d+r0)^2) - K(x,y)/((d+r0)^2) = 0

            T tmp;

            tmp  = diffis;
            tmp *= 0.5;
            scaldiv(tmp,r(0));
            scaldiv(tmp,r(0));
            scaldiv(tmp,r(1));
            tmp += 1.0;

            res = pow(res,-((T) r(1)));

            xygrad  = res;
            xygrad *= -0.5;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            xygrad /= tmp;

//OLD            res = diffis;
//OLD            scaldiv(res,(diffis+r(0)));
//OLD            setnegate(res);
//OLD            res += 1.0;
//OLD
//OLD            xygrad = 0.0;
//OLD
//OLD            diffgrad = res;
//OLD            setnegate(diffgrad);
//OLD            scaldiv(res,(diffis+r(0)));

            break;
        }

        case 9:
        {
            // K = sqrt( d/(r0.r0) + r1^2 )
            // dK = (1/(r0.r0))/(2.K)
            //
            // if d/(r0.r0) + r1^2 = 0 then dk,d2K = 0

            res = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1),r(1));
            OP_sqrt(res);

            if ( (double) abs2(res) > 1e-24 )
            {
                diffgrad  = res;
                diffgrad *= 2.0;
                OP_einv(diffgrad);
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));
            }

            break;
        }

        case 10:
        {
            // K = 1/sqrt( d/(r0.r0) + r1^2 )
            // dK = (1/(r0.r0))*-(K^3)/2
            //
            // Ill-defined as d + r1^2 -> 0

            res = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1),r(1));
            OP_sqrt(res);
            OP_einv(res);

            diffgrad  = res;
            diffgrad *= res;
            diffgrad *= res;
            diffgrad *= -2.0;
            OP_einv(diffgrad);
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            break;
        }

        case 11:
        {
            // K = 2/pi * arccos(-sqrt(d)/r0) - 2/pi * sqrt(d)/r0 * sqrt(1 - d/r0^2)
            // dK = ( 2/pi * 1/sqrt(1 - d/r0^2) - 2/pi * sqrt(1 - d/r0^2) - 1/pi * sqrt(d)/r0 * 1/sqrt(1 - d/r0^2) * -2 sqrt(d)/r0 ) * 1/(2*r0*sqrt(d))
            //    = ( 2/pi * 1/sqrt(1 - d/r0^2) - 2/pi * sqrt(1 - d/r0^2) + 2/pi * d/r0^2 * 1/sqrt(1 - d/r0^2) ) * 1/(2*r0*sqrt(d))
            //    = ( 1/sqrt(1 - d/r0^2) - sqrt(1 - d/r0^2) + d/r0^2 * 1/sqrt(1 - d/r0^2) ) * 1/(pi*r0*sqrt(d))
            //    = ( 1 - 1 + d/r0^2 + d/r0^2 ) * 1/(pi*r0*sqrt(d)*sqrt(1 - d/r0^2))
            //    = ( d/r0^2 + d/r0^2 ) * 1/(pi*r0*sqrt(d)*sqrt(1 - d/r0^2))
            //    = d/r0^2 * 2/(pi*r0*sqrt(d)*sqrt(1 - d/r0^2))
            //    = sqrt(d)/r0^2 * 2/(pi*r0*sqrt(1 - d/r0^2))
            //    = 2/pi 1/r0^2 sqrt(d)/r0 * 1/sqrt(1 - d/r0^2)

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));
            tempres *= -1.0;

            T tempsq(tempres);
            tempsq *= tempsq;
            tempsq *= -1.0;
            tempsq += 1.0;
            OP_sqrt(tempsq);

            if ( (double) abs2(tempres) < 1.0 )
            {
                // K = 2/pi * arccos(-sqrt(d)/r0) - 2/pi * sqrt(d)/r0 * sqrt(1 - d/r0^2)

                res  = tempsq;
                res *= tempres;
                res += acos(tempres);
                res *= NUMBASE_2ONPI;

                //    = 2/pi 1/r0^2 sqrt(d)/r0 * 1/sqrt(1 - d/r0^2)

                diffgrad  = tempsq;
                OP_einv(diffgrad);
                diffgrad *= tempres;
                res *= NUMBASE_2ONPI;
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));
            }

            break;
        }

        case 12:
        {
            // K = 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3
            // dK = ( 3/2 + 3/2 * (sqrt(d)/r0)^2 ) 1/(2*sqrt(d)*r0)
            //    = ( 3/2 + 3/2 * (sqrt(d)/r0)^2 ) 1/2 1/r0^2 r0/sqrt(d)
            //    = 3/4 1/r0^2 ( (sqrt(d)/r0)^{-1} + sqrt(d)/r0 )

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));

            // K = 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3

            res = tempres;
            res *= tempres;
            res *= tempres;
            res /= 2.0;
            res *= 0.6666666666666666666666;
            res -= tempres;
            res /= 0.6666666666666666666666;
            res += 1.0;

            //    = 3/4 1/r0^2 ( (sqrt(d)/r0)^{-1} + sqrt(d)/r0 )

            diffgrad  = tempres;
            OP_einv(diffgrad);
            diffgrad += tempres;
            diffgrad *= 0.75;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            break;
        }

        case 13:
        {
            // K = sinc(sqrt(d)/r0)
            // dK = ( cos(sqrt(d)/r0) - K )/(2*r0*sqrt(d))
            //
            // if d = 0 then dK,d2K = 0

            T tmp(diffis);

            OP_sqrt(tmp);

            T tmpb(tmp);

            scaldiv(tmpb,r(0));

            res = tmpb;
            OP_sinc(res);

            if ( (double) tmpb > 1e-12 )
            {
                diffgrad  = tmpb;
                OP_cos(diffgrad);
                diffgrad -= res;
                diffgrad *= 0.5;
                scaldiv(diffgrad,r(0));
                diffgrad /= tmp;
            }

            break;
        }

        case 14:
        {
            // K = -sqrt(d)^r1
            // dK = -(r1*sqrt(d)^(r1-2))/2 
            //    = -(r1*sqrt(d)^r1)/(2*d) 
            //    = (r1*K)/(2*d) 
            //    = r1*K/(2*d)
            //
            // if d = 0 then dK,d2K = 0

            res  = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            OP_sqrt(res);
            res  = pow(res,(T) r(1));
            res *= -1.0;

            if ( (double) diffis > 1e-12 )
            {
                diffgrad  = res;
                scalmul(diffgrad,r(1));
                diffgrad *= -0.5;
                diffgrad /= diffis;
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));
            }

            break;
        }

        case 15:
        {
            // K = -log(sqrt(d)^r1 + 1)
            // dK = -(r1.sqrt(d)^(r1-2))/(2*(sqrt(d)^r1 + 1))

            T tmpa(diffis);

            scaldiv(tmpa,r(0));
            scaldiv(tmpa,r(0));

            OP_sqrt(tmpa);

            T tmpb;

            tmpb = pow(tmpa,(T) r(1));
            tmpb += 1.0;

            res = tmpb;
            OP_log(res);
            setnegate(res);

            diffgrad = pow(tmpb,((T) r(1))-2.0);
            scalmul(diffgrad,r(1));
            diffgrad *= -0.5;
            diffgrad /= tmpb;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            break;
        }

        case 19:
        {
            // K = 1/(1+(d/(r0.r0)))
            // dK = -K^2/(r0.r0)

            res = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            res += 1.0;
            OP_einv(res);

            diffgrad  = res;
            diffgrad *= res;
            diffgrad *= -1.0;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            break;
        }

        case 23:
        {
            // K = 1/(1+(sqrt(d)/r0)^r1)
            // dK = -(r1/(2.r0)) * (sqrt(d)/r(0))^(r1-2) * K^2

            T tmp(diffis);

            OP_sqrt(tmp);
            scaldiv(tmp,r(0));

            res  = pow(tmp,(T) r(1));
            res += 1.0;
            OP_einv(res);

            T tmpb;
            tmpb = pow(tmp,(T) r(1)-2.0);

            diffgrad  = res;
            diffgrad *= res;
            diffgrad *= -0.5;
            scaldiv(diffgrad,r(0));
            scalmul(diffgrad,r(1));
            diffgrad *= tmpb;

            break;
        }

        case 24:
        {
            // K = (1-(z^i0))/(1-z)
            // dK = ( -i0.(z^(i0-1)) + (1-(z^i0))/(1-z) )/(1-z) 
            //    = ( -i0.(z^i0)/z + K )/(1-z)                         if ( i0 >= 1 )
            //
            // Ill-defined at z = 1
            //
            // z -> z/(r0.r0) and blah blah

            T zsc(xyprod);

            scaldiv(zsc,r(0));
            scaldiv(zsc,r(0));

            if ( ( (double) abs2(zsc) > 1e-12 ) || ( ic(0) == 0 ) )
            {
                T tmp = zsc;

                raiseto(tmp,ic(0));

                res  = 1.0;
                res -= tmp;
                res /= (1.0-zsc);
            }

            if ( ( (double) abs2(zsc) > 1e-12 ) || ( ic(0) == 1 ) )
            {
                T tmp = zsc;

                raiseto(tmp,ic(0)-1);

                xygrad  = tmp;
                xygrad *= -ic(0);
                xygrad += res;
                xygrad /= (1.0-zsc);
                scaldiv(xygrad,r(0));
                scaldiv(xygrad,r(0));
            }

            break;
        }

        case 25:
        {
            // K = pi.cosh(pi-(sqrt(d)/r0))
            // dK = -pi/(2.r0) * sinh(pi-sqrt(d)/r0) / sqrt(d)

            T tmp(diffis);

            OP_sqrt(tmp);

            res  = tmp;
            scaldiv(res,r(0));
            res -= NUMBASE_PI;
            res *= -1.0;
            OP_cosh(res);
            res *= NUMBASE_PI;

            diffgrad  = tmp;
            scaldiv(diffgrad,r(0));
            diffgrad -= NUMBASE_PI;
            diffgrad *= -1.0;
            OP_sinh(diffgrad);
            diffgrad *= NUMBASE_PION2;            
            diffgrad *= -0.5;
            scaldiv(diffgrad,r(0));
            diffgrad /= tmp;

            break;
        }

        case 26:
        {
            // K = ((d/r0)^(r1+0.5))
            // dK = 1/r0 * (r1+0.5) * (d/r0)^(r1-0.5)
            //    = 1/r0 * (r1+0.5) * K/d

            res  = diffis;
            scaldiv(res,r(0));
            res  = pow(res,((T) r(1))+0.5);

            if ( (double) diffis > 1e-12 )
            {
                diffgrad = 0.5;
                scaladd(diffgrad,r(1));
                scaldiv(diffgrad,r(0));
                diffgrad *= res;
                diffgrad /= diffis;
            }

            break;
        }

        case 27:
        {
            // K = ((d/r0)^r1).ln(sqrt(d/r0))
            // dK = (r1/r0).((d/r0)^(r1-1)).ln(sqrt(d/r0)) + ((d/r0)^r1) 1/(sqrt(d/r0)) 1/2 1/sqrt(d*r0)
            //    = (r1/r0).((d/r0)^r1).ln(sqrt(d/r0)).(r0/d) + ((d/r0)^r1)/(2d)
            //    = 2.r1.((d/r0)^r1).ln(sqrt(d/r0))/(2d) + ((d/r0)^r1)/(2d)
            //    = ( 2.r1.ln(sqrt(d/r0)) + 1 ).((d/r0)^r1)/(2d)
            //    = ( ln(sqrt(d/r0)) + 1/(2.r1) ).((d/r0)^r1).(2.r1)/(2d)
            //    = ( ((d/r0)^r1).ln(sqrt(d/r0)) + ((d/r0)^r1)/(2.r1) ).(2.r1)/(2d)
            //    = ( K + ((d/r0)^r1)/(2.r1) ).(2.r1)/(2d)

            T scalres(diffis);

            scaldiv(scalres,r(0));
            OP_sqrt(scalres);

            T tempres(scalres);

            OP_log(tempres);

            scalres = pow(scalres,(T) r(1));

            res  = scalres;
            res *= tempres;

            if ( (double) diffis > 1e-12 )
            {
                diffgrad  = scalres;
                diffgrad *= 0.5;
                scaldiv(diffgrad,r(1));
                diffgrad += res;
                diffgrad *= 2;
                scalmul(diffgrad,r(1));
                diffgrad *= 0.5;
                diffgrad /= diffis;
            }

            break;
        }

        case 32:
        {
            // K = r1 if i == j >= 0, 0 otherwise
            // dK = 0.0

            if ( ( i == j ) && ( i >= 0 ) )
            {
                res = r(1);
            }

            break;
        }

        case 33:
        {
            // K = 1/(2.r0) ( 1 if real(sqrt(d)) < r0, 0 otherwise )
            // dK = 0

            if ( real(sqrt(diffis)-r(0)) < zerogentype() )
            {
                res = 0.5/r(0);
            }

            break;
        }

        case 34:
        {
            // K = (1-sqrt(d)/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )
            // dK = 1/r0 1/(2*sqrt(d)*r0)
            //    = 1/r0 1/2 1/r0^2 r0/sqrt(d)
            //    = 1/r0^3 1/2 r0/sqrt(d)

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));

            if ( (double) real(tempres-1.0) < 0.0 )
            {
                res  = 1.0;
                res -= tempres;
                scaldiv(res,r(0));

                diffgrad = tempres;
                OP_einv(diffgrad);
                diffgrad /= 2.0;
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));
            }

            break;
        }

        case 38:
        {
            // K = exp(-sqrt(d)/r0)
            // dK = 1/r0^2 -r0/sqrt(d) exp(-sqrt(d)/r0)
            //    = 1/r0^2 -r0/sqrt(d) K

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));
            tempres *= -1.0;

            res = tempres;
            OP_exp(res);

            diffgrad  = res;
            diffgrad *= tempres;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            break;
        }

        case 39:
        {
            // K = (1+((sqrt(3)/r0).sqrt(d))) . exp(-(sqrt(3)/r0).sqrt(d))
            // dK = ( exp( -sqrt(3)/r0 sqrt(d) ) - ( 1 + sqrt(3)/r0 sqrt(d) ) exp( -sqrt(3)/r0 sqrt(d) ) ) sqrt(3)/r0 1/2 1/sqrt(d)
            //    = -((sqrt(3)/r0).sqrt(d)) exp(-((sqrt(3)/r0).sqrt(d))) sqrt(3)/r0 1/2 1/sqrt(d)
            //    = -1/2 3/r0^2 exp(-((sqrt(3)/r0).sqrt(d)))
            //    = -3/2 1/r0^2 exp(-((sqrt(3)/r0).sqrt(d)))

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));
            tempres *= sqrt(3.0);

            T expres(tempres);
            expres *= -1.0;
            OP_exp(expres);

            res  = tempres;
            res += 1;
            res *= expres;

            diffgrad  = expres;
            diffgrad *= -1.5;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            break;
        }

        case 42:
        {
            // K = agd(z/(r0.r0))
            // dK = (1/(r0.r0)) sec^2(z/(r0.r0)) / ( 1 - tan^2(z/(r0.r0)) )

            T scalres = diffis;
            scaldiv(scalres,r(0));
            scaldiv(scalres,r(0));

            T scz = scalres;
            OP_sec(scz);
            scz *= scz;

            T taz = scalres;
            OP_tan(taz);

            res = scalres;
            OP_agd(res);

            xygrad  = taz;
            xygrad *= taz;
            xygrad *= -1.0;
            xygrad += 1.0;
            OP_einv(xygrad);
            xygrad *= scz;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 43:
        {
            // K = log((1+r0.z)/(1-r0.z))
            // dK = r0*(1-r0.z)/(1+r0.z)*( 1/(1-r0.z) + (1+r0.z)/(1-r0.z)^2 )
            //    = r0*(1-r0.z)/(1+r0.z)*( (1-r0.z) + (1+r0.z) )/(1-r0.z)^2
            //    = r0*((1-r0.z)/(1+r0.z))/(1-r0.z)^2
            //    = r0/((1+r0.z)*(1-r0.z))
            // ADDENDUM: replace ro with 1/(r0.r0)

            T tempa = xyprod;

            tempa  = xyprod;
            scaldiv(tempa,r(0));
            scaldiv(tempa,r(0));

            T tempb = tempa;

            tempa += 1.0;
            tempb -= 1.0;

            res  = tempa;
            res /= tempb;
            OP_log(res);

            xygrad  = tempa;
            xygrad *= tempb;
            OP_einv(xygrad);
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 44:
        {
            // K = exp(z/(r0.r0))
            // dK = K/(r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scalsub(res,r(1));
            OP_exp(res);

            xygrad = res;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 45:
        {
            // K = sinh(z/(r0.r0))
            // dK = cosh(z/(r0.r0))/(r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            OP_sinh(res);

            xygrad = xyprod;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            OP_cosh(xygrad);
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 46:
        {
            // K = cosh(z/(r0.r0))
            // dK = sinh(z/(r0.r0))/(r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            OP_cosh(res);

            xygrad = xyprod;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            OP_sinh(xygrad);
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 47:
        {
            // K = sinc(sqrt(d)/r0).cos(2*pi*sqrt(d)/(r0.r1))
            //
            // if d = 0 then dK,d2K = 0

            throw("bugger that");

            break;
        }

        case 100:
        {
            // K = z/(r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 103:
        {
            // K = 0 if real(z) < 0, 1 otherwise

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = 0.0;
            }

            else
            {
                res = 1.0;
            }

            xygrad = 0.0;

            break;
        }

        case 104:
        {
            // K = 0 if real(z) < 0, z/(r0*r0) otherwise

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = 0.0;

                xygrad = 0.0;
            }

            else
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));

                xygrad = 1.0;
                scaldiv(xygrad,r(0));
                scaldiv(xygrad,r(0));
            }

            break;
        }

        case 106:
        {
            // K = r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise

            static T zgt(0.0);

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            if ( xyprod < zgt )
            {
                scalmul(res,r(1));
                scalmul(xygrad,r(1));
            }

            break;
        }

        case 200:
        {
            // K = z/(r0.r0) - 1

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            res -= 1.0;

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 203:
        {
            // K = -1 if real(z) < 0, 1 otherwise

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = -1.0;
            }

            else
            {
                res = 1.0;
            }

            xygrad = 0.0;

            break;
        }

        case 204:
        {
            // K = 0 if real(z) < 0, z/(r0*r0) otherwise    - 1

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = -1.0;

                xygrad = 0.0;
            }

            else
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
                res -= 1.0;

                xygrad = 1.0;
                scaldiv(xygrad,r(0));
                scaldiv(xygrad,r(0));
            }

            break;
        }

        case 206:
        {
            // K = r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise

            static T zgt(0.0);

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            if ( xyprod < zgt )
            {
                scalmul(res,r(1));
                scalmul(xygrad,r(1));
            }

            res -= 1.0;

            break;
        }

        default:
        {
            throw("fee fi fo fum");

            break;
        }
    }

    res      *= (const T &) weight;
    xygrad   *= (const T &) weight;
    diffgrad *= (const T &) weight;

    return;
}

template <class T>
void MercerKernel::d2KKpro(T &totxygrad, T &totxnormgrad, T &totxyxygrad,
                           T &totxyxnormgrad, T &totxyynormgrad,
                           T &totxnormxnormgrad, T &totxnormynormgrad, T &totynormynormgrad,
                           T &totres,
                           const T &inxyprod, const T &diffis,
                           int i, int j,
                           int locindstart, int locindend,
                           int xdim, int m, const T &xxprod, const T &yyprod) const
{
    T xyprod = inxyprod;

    totres = xyprod;

    totxygrad    = 1.0;
    totxnormgrad = 0.0;

    totxyxygrad       = 0.0;
    totxyxnormgrad    = 0.0;
    totxyynormgrad    = 0.0;
    totxnormxnormgrad = 0.0;
    totxnormynormgrad = 0.0;
    //totynormxnormgrad = 0.0;
    totynormynormgrad = 0.0;

    if ( !isFastKernelSum() && ( locindstart > locindend ) )
    {
        return;
    }

    // Base kernel is: kn(k{n-1}(...k1(k0(q))...)), where q is either <x,y> or ||x-y||^2
    //
    // Chained derivative is the prod: dkn(k{n-1}(...k1(k0(q)))) * dk{n-1}(...k1(k0(q)))... * dk1(k0(q)) * dk0(q)
    //
    // Chained 2nd derivative is sum/prod:   d2kn(k{n-1}(...k1(k0(q)))) * dk{n-1}(...k1(k0(q)))... * dk1(k0(q)) * dk0(q)
    //                                     + dkn(dk{n-1(...k1(k0(q)))) * d2k{n-1}(...k1(k0(q)))... * dk1(k0(q)) * dk0(q)
    //                                     + dkn(dk{n-1(...k1(k0(q)))) * dk{n-1}(...k1(k0(q)))... * d2k1(k0(q)) * dk0(q)
    //                                     + dkn(dk{n-1(...k1(k0(q)))) * dk{n-1}(...k1(k0(q)))... * dk1(k0(q)) * d2k0(q)

    T totdiffgrad;
    T totdiffdiffgrad;

    totdiffgrad     = 0.0;
    totdiffdiffgrad = 0.0;

    int ind;

    // NB: xyprod is used at all layers for chained kernels
    //     diffis is used only at first layer for chained kernels

    Vector<T> allres(locindend-locindstart+1);

    Vector<T> allxygrad(locindend-locindstart+1);
    Vector<T> alldiffgrad(locindend-locindstart+1);
    Vector<T> allxnormonlygrad(locindend-locindstart+1);

    Vector<T> allxyxygrad(locindend-locindstart+1);
    Vector<T> alldiffdiffgrad(locindend-locindstart+1);
    Vector<T> allxnormxnormonlygrad(locindend-locindstart+1);
    Vector<T> allxnormynormonlygrad(locindend-locindstart+1);
    //Vector<T> allynormxnormonlygrad(locindend-locindstart+1);
    Vector<T> allynormynormonlygrad(locindend-locindstart+1);

    for ( ind = locindstart ; ind <= locindend ; ind++ )
    {
        T &res = allres("&",ind-locindstart);

        T &xygrad        = allxygrad("&",ind-locindstart);
        T &diffgrad      = alldiffgrad("&",ind-locindstart);
        T &xnormonlygrad = allxnormonlygrad("&",ind-locindstart);

        T &xyxygrad           = allxyxygrad("&",ind-locindstart);
        T &diffdiffgrad       = alldiffdiffgrad("&",ind-locindstart);
        T &xnornxnormonlygrad = allxnormxnormonlygrad("&",ind-locindstart);
        T &xnornynormonlygrad = allxnormynormonlygrad("&",ind-locindstart);
        //T &ynornxnormonlygrad = allynormxnormonlygrad("&",ind-locindstart);
        T &ynornynormonlygrad = allynormynormonlygrad("&",ind-locindstart);

        retVector<gentype> tmpva;

        const Vector<gentype> &r = dRealConstants(ind)(1,1,dRealConstants(ind).size()-1,tmpva);
        const Vector<int> &ic = dIntConstants(ind);

        int ktype = cType(ind);

        d2KKprosingle(xygrad,diffgrad,xnormonlygrad,xyxygrad,diffdiffgrad,xnornxnormonlygrad,xnornynormonlygrad,ynornynormonlygrad,res,xyprod,diffis,i,j,xdim,m,xxprod,yyprod,ktype,cWeight(locindstart),r,ic,isMagTerm(ind));

        if ( isFastKernelSum() && ( ind == locindstart ) )
        {
            totres = res;

            totxygrad    = xygrad;
            totdiffgrad  = diffgrad;
            totxnormgrad = xnormonlygrad;

            totxyxygrad       = xyxygrad;
            totdiffdiffgrad   = diffdiffgrad;
            totxnormxnormgrad = xnornxnormonlygrad;
            totxnormynormgrad = xnornynormonlygrad;
            //totynormxnormgrad = ynornxnormonlygrad;
            totynormynormgrad = ynornynormonlygrad;
        }

        else if ( isFastKernelSum() )
        {
            totres += res;

            totxygrad    += xygrad;
            totdiffgrad  += diffgrad;
            totxnormgrad += xnormonlygrad;

            totxyxygrad       += xyxygrad;
            totdiffdiffgrad   += diffdiffgrad;
            totxnormxnormgrad += xnornxnormonlygrad;
            totxnormynormgrad += xnornynormonlygrad;
            //totynormxnormgrad += ynornxnormonlygrad;
            totynormynormgrad += ynornynormonlygrad;
        }

        else if ( ( ind == locindend ) && ( ind == locindstart ) )
        {
            totres = res;

            totxygrad    = xygrad;
            totdiffgrad  = diffgrad;
            totxnormgrad = xnormonlygrad;

            totxyxygrad       = xyxygrad;
            totdiffdiffgrad   = diffdiffgrad;
            totxnormxnormgrad = xnornxnormonlygrad;
            totxnormynormgrad = xnornynormonlygrad;
            //totynormxnormgrad = ynornxnormonlygrad;
            totynormynormgrad = ynornynormonlygrad;
        }

        else if ( ind == locindstart )
        {
            xyprod = res;

            totxygrad    = xygrad;
            totdiffgrad  = diffgrad;
            totxnormgrad = xnormonlygrad;

            // second-order gradients need to be post-calculated
        }

        else if ( ind < locindend )
        {
            xyprod = res;

            totxygrad    *= xygrad;
            totdiffgrad  *= xygrad; // diffgrad zero here by definition
            totxnormgrad *= xnormonlygrad; // likewise

            // second-order gradients need to be post-calculated
        }

        else
        {
            totres = res;

            totxygrad    *= xygrad;
            totdiffgrad  *= xygrad; // diffgrad zero here by definition
            totxnormgrad *= xnormonlygrad; // likewise

            // second-order gradients need to be post-calculated
        }
    }

    // second-oder gradient post-calculation

    if ( !isFastKernelSum() && ( locindend > locindstart ) )
    {
        // So *not* magterm, meaning totxnormgrad = 0 at this point

        totxyxygrad     = 0.0;
        totdiffdiffgrad = 0.0;

        int indin;

        for ( ind = locindstart ; ind <= locindend ; ind++ )
        {
            T tempxyxygrad;     tempxyxygrad     = 1.0;
            T tempdiffdiffgrad; tempdiffdiffgrad = 1.0;

            for ( indin = locindstart ; indin <= locindend ; indin++ )
            {
                if ( indin == ind )
                {
                    T &xyxygrad     = allxyxygrad("&",indin-locindstart);
                    T &diffdiffgrad = alldiffdiffgrad("&",indin-locindstart);

                    tempxyxygrad     *= xyxygrad;
                    tempdiffdiffgrad *= diffdiffgrad;
                }

                else
                {
                    T &xygrad   = allxygrad("&",indin-locindstart);
                    T &diffgrad = alldiffgrad("&",indin-locindstart);

                    tempxyxygrad     *= xygrad;
                    tempdiffdiffgrad *= diffgrad;
                }
            }

            totxyxygrad     += tempxyxygrad;
            totdiffdiffgrad += tempdiffdiffgrad;
        }
    }

    // diffis = ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    //
    // dK/dxyprod = partialK/partialxyprod + partialK/partialdiffis partialdiffis/partialxyprod
    //            = partialK/partialxyprod - 2*partialK/partialdiffis
    // dK/dxnorm  = partialK/partialdiffis partialdiffis/partialxnorm
    //            = partialK/partialdiffis 

    totxygrad -= totdiffgrad;
    totxygrad -= totdiffgrad;

    totxnormgrad += totdiffgrad;

    // diffis = ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    //
    // dK/dxyprod = partialK/partialxyprod + partialK/partialdiffis partialdiffis/partialxyprod
    //            = partialK/partialxyprod - 2*partialK/partialdiffis
    // dK/dxnorm  = partialK/partialdiffis partialdiffis/partialxnorm
    //            = partialK/partialdiffis 
    // dK/dynorm  = partialK/partialdiffis partialdiffis/partialynorm
    //            = partialK/partialdiffis 
    //
    // d2K/dxyxy = partial2K/partialxyxy - 2*partial2K/partialdiffdiffis partialdiffis/partialxyprod
    //           = partial2K/partialxyxy + 4*partial2K/partialdiffdiffis
    // d2K/dxyxnorm = partial2K/partialdiffdiffis partialdiffis/partialxyprod
    //              = -2*partial2K/partialdiffdiffis
    // d2K/dxyynorm = partial2K/partialdiffdiffis partialdiffis/partialxyprod
    //              = -2*partial2K/partialdiffdiffis
    // d2K/dxnormxnorm = partial2K/partialdiffdiffis partialdiffis/partialxnorm
    //                 = partial2K/partialdiffdiffis
    // d2K/dxnormynorm = partial2K/partialdiffdiffis partialdiffis/partialynorm
    //                 = partial2K/partialdiffdiffis
    // d2K/dynormxnorm = partial2K/partialdiffdiffis partialdiffis/partialxnorm
    //                 = partial2K/partialdiffdiffis
    // d2K/dynormynorm = partial2K/partialdiffdiffis partialdiffis/partialynorm
    //                 = partial2K/partialdiffdiffis

    totxyxygrad += totdiffdiffgrad;
    totxyxygrad += totdiffdiffgrad;
    totxyxygrad += totdiffdiffgrad;
    totxyxygrad += totdiffdiffgrad;

    totxyxnormgrad  = totdiffdiffgrad;
    totxyxnormgrad += totdiffdiffgrad;
    totxyxnormgrad *= -1.0;

    totxyynormgrad  = totdiffdiffgrad;
    totxyynormgrad += totdiffdiffgrad;
    totxyynormgrad *= -1.0;

    totxnormxnormgrad += totdiffdiffgrad;
    totxnormynormgrad += totdiffdiffgrad;
//    totynormxnormgrad += totdiffdiffgrad;
    totynormynormgrad += totdiffdiffgrad;

    return;
}



//KERNELSHERE
//phantomx

template <class T>
void MercerKernel::d2KKprosingle(T &xygrad, T &diffgrad, T &xnormonlygrad,
                                 T &xyxygrad, T &diffdiffgrad,
                                 T &xnormxnormonlygrad, T &xnormynormonlygrad, T &ynormynormonlygrad,
                                 T &res,
                                 const T &xyprod, const T &diffis,
                                 int i, int j,
                                 int xdim, int m,
                                 const T &xxprod, const T &yyprod,
                                 int ktype, const gentype &weight,
                                 const Vector<gentype> &r, const Vector<int> &ic,
                                 int magterm) const
{
    res = 0.0;

    xygrad        = 0.0;
    diffgrad      = 0.0;
    xnormonlygrad = 0.0;

    xyxygrad     = 0.0;
    diffdiffgrad = 0.0;

    if ( magterm )
    {
        xygrad = 0.0;

        T xres; xres = 0.0;
        T yres; yres = 0.0;

        T ddgrad;         ddgrad         = 0.0;
        T ddiffgrad;      ddiffgrad      = 0.0;
        T ddnormonlygrad; ddnormonlygrad = 0.0;

        T ddiffdiffgrad;      ddiffdiffgrad      = 0.0;
        T dnormdnormonlygrad; dnormdnormonlygrad = 0.0;

        T altdiffis; altdiffis = 0.0;

        gentype altweight(1.0);

        T xxgrad; xxgrad = 0.0;
        T yygrad; yygrad = 0.0;

        T xxxxgrad; xxxxgrad = 0.0;
        T yyyygrad; yyyygrad = 0.0;

        d2KKprosingle(xxgrad,ddiffgrad,ddnormonlygrad,xxxxgrad,ddiffdiffgrad,dnormdnormonlygrad,dnormdnormonlygrad,dnormdnormonlygrad,xres,xxprod,altdiffis,i,i,xdim,m,xxprod,xxprod,ktype,altweight,r,ic,0);
        d2KKprosingle(yygrad,ddiffgrad,ddnormonlygrad,yyyygrad,ddiffdiffgrad,dnormdnormonlygrad,dnormdnormonlygrad,dnormdnormonlygrad,yres,yyprod,altdiffis,j,j,xdim,m,yyprod,yyprod,ktype,altweight,r,ic,0);

        (void) ddgrad;
        (void) ddiffgrad;
        (void) ddnormonlygrad;

        (void) ddiffdiffgrad;
        (void) dnormdnormonlygrad;

        res  = xres;
        res *= yres;

        //xygrad  = 0.0;
        //xygrad *= yres;
        
        //diffgrad  = 0.0;
        //diffgrad *= yres;

        xnormonlygrad  = xxgrad;
        xnormonlygrad *= yres;

        //xyxygrad  = 0.0;
        //xyxygrad *= yres;

        //diffdiffgrad  = 0.0;
        //diffdiffgrad *= yres;

        xnormxnormonlygrad  = xxxxgrad;
        xnormxnormonlygrad *= yres;

        xnormynormonlygrad  = xxgrad;
        xnormynormonlygrad *= yygrad;

        //ynormxnormonlygrad  = ??grad;
        //ynormxnormonlygrad *= ??grad;

        ynormynormonlygrad  = xres;
        ynormynormonlygrad *= yyyygrad;

        res                *= (const T &) weight;
        //xygrad             *= (const T &) weight;
        //diffgrad           *= (const T &) weight;
        xnormonlygrad      *= (const T &) weight;
        xyxygrad           *= (const T &) weight;
        diffdiffgrad       *= (const T &) weight;
        xnormxnormonlygrad *= (const T &) weight;
        xnormynormonlygrad *= (const T &) weight;
        //ynormxnormonlygrad *= (const T &) weight;
        ynormynormonlygrad *= (const T &) weight;

        return;
    }

    switch ( ktype )
    {
        case 0:
        {
            // K = r1
            // dK = 0
            // d2K = 0

            res = r(1);

            break;
        }

        case 1:
        {
            // K = z/(r0.r0)
            // dK = 1/(r0.r0)
            // d2K = 0

            res    = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 2:
        {
            // K = ( r1 + z/(r0.r0) )^i0
            // dK = i0/(r0.r0) * ( r1 + z/(r0.r0) )^(i0-1)               if ( i0 >= 1 )
            // d2K = i0.(i0-1)/(r0.r0.r0.r0) * ( r1 + z/(r0.r0) )^(i0-2) if ( i0 >= 2 )

            T temp(xyprod);

            scaldiv(temp,r(0));
            scaldiv(temp,r(0));
            scaladd(temp,r(1));

            res = temp;
            raiseto(res,ic(0));

            if ( ic(0) >= 1 )
            {
                xygrad = temp;
                raiseto(xygrad,ic(0)-1);
                xygrad *= ic(0);
                scaldiv(xygrad,r(0));
                scaldiv(xygrad,r(0));
            }

            if ( ic(0) >= 2 )
            {
                xyxygrad = temp;
                raiseto(xyxygrad,ic(0)-2);
                xyxygrad *= ic(0)*(ic(0)-1);
                scaldiv(xyxygrad,r(0));
                scaldiv(xyxygrad,r(0));
                scaldiv(xyxygrad,r(0));
                scaldiv(xyxygrad,r(0));
            }

            break;
        }

        case 3:
        {
            // K = exp(-d/(2.r0.r0))
            // dK = -K/(2*r0*r0)
            // d2K = -dK/(2*r0*r0)

            res  = diffis;
            res *= -0.5;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            res += log(AltDiffNormConst(xdim,m,r(0)));
            scalsub(res,r(1));
            OP_exp(res);

            diffgrad  = res;
            diffgrad *= -0.5;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            diffdiffgrad  = diffgrad;
            diffdiffgrad *= -0.5;
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));

            break;
        }

        case 4:
        {
            // K = exp(-sqrt(d)/r0)
            // dK = -K/(2*r0*sqrt(d))
            // d2K = -dK/(2*r0*sqrt(d)) + K/(4*r0*d*sqrt(d))
            //     = -dK/(2*r0*sqrt(d)) - dK/(2*d)
            //     = -( 1/(r0*sqrt(d)) + 1/d )*dK/2
            //     = -( sqrt(d)/r0 + 1 )*dK/(2.d)
            //
            // At d=0: dK = 0, d2K = 0

            T tmp(diffis);

            OP_sqrt(tmp);

            res = tmp;
            setnegate(res);
            scaldiv(res,r(0));
            scalsub(res,r(1));
            OP_exp(res);

            if ( (double) abs2(diffis) > 1e-12 )
            {
                diffgrad = res;
                diffgrad *= -0.5;
                scaldiv(diffgrad,r(0));
                diffgrad /= tmp;

                diffdiffgrad = tmp;
                scaldiv(diffdiffgrad,r(0));
                diffdiffgrad += 1.0;
                diffdiffgrad *= diffgrad;
                diffdiffgrad *= -0.5;
                scaldiv(diffdiffgrad,diffis);
            }

            break;
        }

        case 5:
        {
            // K = exp(-sqrt(d)^r1/(r1*r0^r1))
            // dK = -K*((sqrt(d)^(r1-2))/(2*r0^r1))
            //    = -K*((sqrt(d)^r1)/(2*d*r0^r1))
            // d2K = -dK*((sqrt(d)^(r1-2))/(2*r0^r1)) - K*(r1-2)*((sqrt(d)^(r1-4))/(4*r0^r1))
            //     = -dK*((sqrt(d)^(r1-2))/(2*r0^r1)) - (r1-2)*K*((sqrt(d)^(r1-2))/(4*d*r0^r1))
            //     = -dK*((sqrt(d)^(r1-2))/(2*r0^r1)) + (r1-2)*dK/(2*d)
            //     = -( ((sqrt(d)^(r1-2))/(r0^r1)) - (r1-2)/d )*dK/2
            //     = -( ((sqrt(d)^r1)/(d*r0^r1)) - (r1-2)/d )*dK/2
            //     = -( ((sqrt(d)^r1)/(r0^r1)) - (r1-2) )*dK/(2*d)
            //     = ( -((sqrt(d)^r1)/(r0^r1)) + (r1-2) )*dK/(2*d)
            //
            // At d=0, if r1 < 2: dK = 0, d2K = 0

            T tmpb = pow(sqrt(diffis),(T) r(1));
            T tmpc = pow((T) r(0),(T) r(1));

            res  = tmpb;
            res /= tmpc;
            scaldiv(res,r(1));
            setnegate(res);
            scalsub(res,r(2));
            OP_exp(res);

            if ( ( (double) abs2(diffis) > 1e-12 ) || ( (double) abs2(r(1)) >= 2.0 ) )
            {
                // dK = -K*((sqrt(d)^r1)/(2*d*r0^r1))

                diffgrad  = res;
                diffgrad *= tmpb;
                diffgrad *= -0.5;
                diffgrad /= diffis;
                diffgrad /= tmpc;

                // d2K = ( -((sqrt(d)^r1)/(r0^r1)) + (r1-2) )*dK/(2*d)

                diffdiffgrad  = tmpb;
                diffdiffgrad /= tmpc;
                diffdiffgrad *= -1.0;
                scaladd(diffdiffgrad,r(1));
                diffdiffgrad -= 2.0;
                diffdiffgrad *= diffgrad;
                diffdiffgrad *= 0.5;
                diffdiffgrad /= diffis;
            }

            break;
        }

        case 7:
        {
            // K = tanh( z/(r0.r0) + r1 )
            // dK = 1/(r0.r0) * sech^2( z/(r0.r0) + r1 )
            // d2K = -2.sech( z/(r0.r0) + r1 ).sech( z/(r0.r0) + r1).tanh( z/(r0.r0) + r1)/(r0.r0.r0.r0)
            //     = -2.sech^2( z/(r0.r0) + r1 ).tanh( z/(r0.r0) + r1)/(r0.r0.r0.r0)
            //     = -2.dK.K/(r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1));
            OP_tanh(res);

            xygrad = xyprod;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            scaladd(xygrad,r(1));
            OP_sech(xygrad);
            xygrad *= xygrad;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            xyxygrad  = xygrad;
            xyxygrad *= res;
            scaldiv(xyxygrad,r(0));
            scaldiv(xyxygrad,r(0));
            xyxygrad *= -2.0;

            break;
        }

        case 8:
        {
            // K = ( 1 + d/(2*r0*r0*r1) )^(-r1)
            // dK = -(( 1 + d/(2*r0*r0*r1) )^(-r1-1))/(2*r0*r0)
            //    = -K/(2*r0*r0*( 1 + d/(2*r0*r0*r1) ))
            // d2K = (r1-1).(( 1 + d/(2*r0*r0*r1) )^(-r1-2))/(4*r0*r0*r0*r0*r1)
            //     = (r1-1).(( 1 + d/(2*r0*r0*r1) )^(-r1))/(2*r0*r0*( 1 + d/(2*r0*r0*r1) )*2*r0*r0*( 1 + d/(2*r0*r0*r1) )*r1)
            //     = -(r1-1).dK/(2*r0*r0*( 1 + d/(2*r0*r0*r1) )*r1)

            //OLD     8  | Rational quadratic     | 1 - d/(d+r0)
            //OLD     8  | Rational quadratic     | -1/(d+r0) + d/(d+r0)^2 = -K(x,y)/(d+r0)
            //OLD     8  | Rational quadratic     | K(x,y)/((d+r0)^2) - K(x,y)/((d+r0)^2) = 0

            T tmp;

            tmp  = diffis;
            tmp *= 0.5;
            scaldiv(tmp,r(0));
            scaldiv(tmp,r(0));
            scaldiv(tmp,r(1));
            tmp += 1.0;

            res = pow(res,-((T) r(1)));

            xygrad  = res;
            xygrad *= -0.5;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            xygrad /= tmp;

            xyxygrad  = xygrad;
            xyxygrad *= ((T) res)-1.0;
            xyxygrad *= -0.5;
            scaldiv(xyxygrad,r(0));
            scaldiv(xyxygrad,r(0));
            scaldiv(xyxygrad,r(1));
            xyxygrad /= tmp;

//OLD            res = diffis;
//OLD            scaldiv(res,(diffis+r(0)));
//OLD            setnegate(res);
//OLD            res += 1.0;
//OLD
//OLD            xygrad = 0.0;
//OLD
//OLD            diffgrad = res;
//OLD            setnegate(diffgrad);
//OLD            scaldiv(res,(diffis+r(0)));

            break;
        }

        case 9:
        {
            // K = sqrt( d + r1^2 )
            // dK = 1/(2.K)
            // d2K = -1/(4.K^3)
            //
            // if d + r1^2 = 0 then dk,d2K = 0
            //
            // d -> d/(r0.r0) blah blah

            res = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1),r(1));
            OP_sqrt(res);

            if ( (double) abs2(res) > 1e-24 )
            {
                diffgrad  = res;
                diffgrad *= 2.0;
                OP_einv(diffgrad);
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));

                diffdiffgrad  = res;
                diffdiffgrad *= res;
                diffdiffgrad *= res;
                diffdiffgrad *= 4.0;
                OP_einv(diffdiffgrad);
                diffdiffgrad *= -1.0;
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
            }

            break;
        }

        case 10:
        {
            // K = 1/sqrt( d + r1^2 )
            // dK = -(K^3)/2
            // d2K = 3.(K^5)/4
            //
            // Ill-defined as d + r0^2 -> 0
            //
            // d -> d/(r0.r0) blah blah

            res = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scaladd(res,r(1),r(1));
            OP_sqrt(res);
            OP_einv(res);

            diffgrad  = res;
            diffgrad *= res;
            diffgrad *= res;
            diffgrad *= -2.0;
            OP_einv(diffgrad);
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            diffdiffgrad  = res;
            diffdiffgrad *= res;
            diffdiffgrad *= res;
            diffdiffgrad *= res;
            diffdiffgrad *= res;
            diffdiffgrad *= 4.0;
            OP_einv(diffdiffgrad);
            diffdiffgrad *= 3.0;
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));

            break;
        }

        case 11:
        {
            // K = 2/pi * arccos(-sqrt(d)/r0) - 2/pi * sqrt(d)/r0 * sqrt(1 - d/r0^2)
            // dK = ( 2/pi * 1/sqrt(1 - d/r0^2) - 2/pi * sqrt(1 - d/r0^2) - 1/pi * sqrt(d)/r0 * 1/sqrt(1 - d/r0^2) * -2 sqrt(d)/r0 ) * 1/(2*r0*sqrt(d))
            //    = ( 2/pi * 1/sqrt(1 - d/r0^2) - 2/pi * sqrt(1 - d/r0^2) + 2/pi * d/r0^2 * 1/sqrt(1 - d/r0^2) ) * 1/(2*r0*sqrt(d))
            //    = ( 1/sqrt(1 - d/r0^2) - sqrt(1 - d/r0^2) + d/r0^2 * 1/sqrt(1 - d/r0^2) ) * 1/(pi*r0*sqrt(d))
            //    = ( 1 - 1 + d/r0^2 + d/r0^2 ) * 1/(pi*r0*sqrt(d)*sqrt(1 - d/r0^2))
            //    = ( d/r0^2 + d/r0^2 ) * 1/(pi*r0*sqrt(d)*sqrt(1 - d/r0^2))
            //    = d/r0^2 * 2/(pi*r0*sqrt(d)*sqrt(1 - d/r0^2))
            //    = sqrt(d)/r0^2 * 2/(pi*r0*sqrt(1 - d/r0^2))
            //    = 2/pi 1/r0^2 sqrt(d)/r0 * 1/sqrt(1 - d/r0^2)
            // d2K = 2/pi 1/r0^2 ( 1/sqrt(1 - d/r0^2) - 1/2 sqrt(d)/r0 1/(1 - d/r0^2)^(3/2) -2 sqrt(d)/r0 ) 1/(2*r0*sqrt(d))
            //     = 2/pi 1/r0^2 ( 1 - d/r0^2 + d/r0^2 ) 1/(2*r0*sqrt(d)) 1/(1 - d/r0^2) 1/sqrt(1 - d/r0^2)
            //     = 2/pi 1/r0^2 1/(2*r0*sqrt(d)) 1/(1 - d/r0^2) 1/sqrt(1 - d/r0^2)
            //     = 2/pi 1/r0^2 1/(2*r0^2*sqrt(d)/r0) 1/(1 - d/r0^2) 1/sqrt(1 - d/r0^2)
            //     = 1/pi 1/r0^4 1/(sqrt(d)/r0) 1/(1 - d/r0^2) 1/sqrt(1 - d/r0^2)

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));
            tempres *= -1.0;

            T tempsq(tempres);
            tempsq *= tempsq;
            tempsq *= -1.0;
            tempsq += 1.0;
            OP_sqrt(tempsq);

            if ( (double) abs2(tempres) < 1.0 )
            {
                // K = 2/pi * arccos(-sqrt(d)/r0) - 2/pi * sqrt(d)/r0 * sqrt(1 - d/r0^2)

                res  = tempsq;
                res *= tempres;
                res += acos(tempres);
                res *= NUMBASE_2ONPI;

                //    = 2/pi 1/r0^2 sqrt(d)/r0 * 1/sqrt(1 - d/r0^2)

                diffgrad  = tempsq;
                OP_einv(diffgrad);
                diffgrad *= tempres;
                res *= NUMBASE_2ONPI;
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));

                //     = 1/pi 1/r0^4 1/(sqrt(d)/r0) 1/sqrt(1 - d/r0^2)^3

                diffdiffgrad  = tempsq;
                diffdiffgrad *= tempsq;
                diffdiffgrad *= tempsq;
                diffdiffgrad /= tempres;
                diffdiffgrad /= NUMBASE_PI;
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
            }

            break;
        }

        case 12:
        {
            // K = 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3
            // dK = ( 3/2 + 3/2 * (sqrt(d)/r0)^2 ) 1/(2*sqrt(d)*r0)
            //    = ( 3/2 + 3/2 * (sqrt(d)/r0)^2 ) 1/2 1/r0^2 r0/sqrt(d)
            //    = 3/4 1/r0^2 ( (sqrt(d)/r0)^{-1} + sqrt(d)/r0 )
            // d2K = 3/4 1/r0^2 ( 1 - (sqrt(d)/r0)^{-2} ) 1/(2*sqrt(d)*r0)
            //     = 3/4 1/r0^2 ( 1 - (sqrt(d)/r0)^{-2} ) 1/2 1/r0^2 r0/sqrt(d)
            //     = 3/8 1/r0^4 ( (sqrt(d)/r0)^{-1} - (sqrt(d)/r0)^{-3} )

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));

            // K = 1 - 3/2 * sqrt(d)/r0 + 1/2 * sqrt(d)^3/r0^3

            res = tempres;
            res *= tempres;
            res *= tempres;
            res /= 2.0;
            res *= 0.6666666666666666666666;
            res -= tempres;
            res /= 0.6666666666666666666666;
            res += 1.0;

            //    = 3/4 1/r0^2 ( (sqrt(d)/r0)^{-1} + sqrt(d)/r0 )

            diffgrad  = tempres;
            OP_einv(diffgrad);
            diffgrad += tempres;
            diffgrad *= 0.75;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            //     = 3/8 1/r0^4 ( (sqrt(d)/r0)^{-1} - (sqrt(d)/r0)^{-3} )

            OP_einv(tempres);
            diffdiffgrad  = tempres;            
            diffdiffgrad *= tempres;            
            diffdiffgrad -= 1.0;            
            diffdiffgrad *= tempres;            
            diffdiffgrad *= -1.0;            
            diffdiffgrad *= 3.0/8.0;
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));

            break;
        }

        case 13:
        {
            // K = sinc(sqrt(d)/r0)
            // dK = ( cos(sqrt(d)/r0) - K )/(2*r0*sqrt(d))
            // d2K = ( -sin(sqrt(d)/r0)/(2*r0*sqrt(d)) - dK )/(2*r0*sqrt(d)) - ( cos(sqrt(d)/r0) - K )/(4*r0*d*sqrt(d))
            //     = ( -K/(2*r0^2) - dK )/(2*r0*sqrt(d)) - dK/(2*d)
            //     = ( -K/(2*r0^2) - dK )/(2*r0*sqrt(d)) - ( dK*(2*r0*sqrt(d))/(2*d) )/(2*r0*sqrt(d))
            //     = ( -K/(2*r0^2) - dK*( 1 + r0/sqrt(d) ) )/(2*r0*sqrt(d))
            //     = ( -K/(2*r0^2) - dK*( sqrt(d) + r0 )/sqrt(d) )/(2*r0*sqrt(d))
            //     = ( -K - dK*(2*r0^2)*( sqrt(d) + r0 )/sqrt(d) )/(4*(r0^3)*sqrt(d))
            //     = -( K + dK*(2*r0^2)*( sqrt(d) + r0 )/sqrt(d) )/(4*(r0^3)*sqrt(d))
            //
            // if d = 0 then dK,d2K = 0

            T tmp(diffis);

            OP_sqrt(tmp);

            T tmpb(tmp);

            scaldiv(tmpb,r(0));

            res = tmpb;
            OP_sinc(res);

            if ( (double) tmpb > 1e-12 )
            {
                diffgrad  = tmpb;
                OP_cos(diffgrad);
                diffgrad -= res;
                diffgrad *= 0.5;
                scaldiv(diffgrad,r(0));
                diffgrad /= tmp;

                diffdiffgrad  = tmp;
                scaladd(diffdiffgrad,r(0));
                diffdiffgrad *= diffgrad;
                diffdiffgrad *= 2.0;
                scalmul(diffdiffgrad,r(0));
                scalmul(diffdiffgrad,r(0));
                diffdiffgrad /= tmp;
                diffdiffgrad += res;
                diffgrad *= -0.25;
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
                diffdiffgrad /= tmp;
            }

            break;
        }

        case 14:
        {
            // K = -sqrt(d)^r1
            // dK = -(r1*sqrt(d)^(r1-2))/2 
            //    = -(r1*sqrt(d)^r1)/(2*d) 
            //    = (r1*K)/(2*d) 
            //    = r1*K/(2*d)
            // d2K = -(r1*(r1-2)*sqrt(d)^(r1-4))/4
            //     = -(r1*(r1-2)*sqrt(d)^r1)/(4*d^2)
            //     = (r1*(r1-2)*K)/(4*d^2)
            //     = (r1-2)*dK/(2*d)
            //
            // if d = 0 then dK,d2K = 0

            res  = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            OP_sqrt(res);
            res  = pow(res,(T) r(1));
            res *= -1.0;

            if ( (double) abs2(diffis) > 1e-12 )
            {
                diffgrad  = res;
                scalmul(diffgrad,r(1));
                diffgrad *= -0.5;
                diffgrad /= diffis;
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));

                diffdiffgrad  = diffgrad;
                scalmul(diffdiffgrad,((T) r(1))-2.0);
                diffdiffgrad *= -0.5;
                diffdiffgrad /= diffis;
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
            }

            break;
        }

        case 15:
        {
            // K = -log(sqrt(d)^r1 + 1)
            // dK = -(r1.sqrt(d)^(r1-2))/(2*(sqrt(d)^r1 + 1))
            // d2K = -(r1.(r1-2).sqrt(d)^(r1-4))/(4.(sqrt(d)^r1 + 1)) + ((r1.sqrt(d)^(r1-2))^2)/((2*(sqrt(d)^r1 + 1))^2)
            //     = -(r1-2)*dK/(2*d) + dK^2
            //     = dK*( dK - (r1-2)/(2*d) )

            T tmpa(diffis);

            scaldiv(tmpa,r(0));
            scaldiv(tmpa,r(0));

            OP_sqrt(tmpa);

            T tmpb;

            tmpb = pow(tmpa,(T) r(1));
            tmpb += 1.0;

            res = tmpb;
            OP_log(res);
            setnegate(res);

            diffgrad = pow(tmpb,((T) r(1))-2.0);
            scalmul(diffgrad,r(1));
            diffgrad *= -0.5;
            diffgrad /= tmpb;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            diffdiffgrad = (T) r(1);
            diffdiffgrad -= 2.0;
            diffdiffgrad *= -0.5;
            diffdiffgrad /= diffis;
            diffdiffgrad += diffgrad;
            diffdiffgrad *= diffgrad;
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));

            break;
        }

        case 19:
        {
            // K = 1/(1+(d/(r0.r0)))
            // dK = -K^2/(r0.r0)
            // d2K = 2*K^3/r0^4
            //     = -2*dK*K/(r0*r0)

            res = diffis;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            res += 1.0;
            OP_einv(res);

            diffgrad  = res;
            diffgrad *= res;
            diffgrad *= -1.0;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            diffdiffgrad = diffgrad;
            diffdiffgrad *= res;
            diffdiffgrad *= -2.0;
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));

            break;
        }

        case 23:
        {
            // K = 1/(1+(sqrt(d)/r0)^r1)
            // dK = -r1/(2.r0) * (sqrt(d)/r0)^(r1-2) * K^2
            // d2K = ( -r1/(4.r0^2) * (r1-2) * (sqrt(d)/r0)^(r1-4) * K^2 ) + ( -r1/(2.r0) * (sqrt(d)/r0)^(r1-2) * 2 * K * dK )
            //     = ( -r1/(2.r0^2) * (sqrt(d)/r0)^(r1-2) * K^2 * (r1-2)/(2*d/r0) ) + ( -r1/(2.r0) * (sqrt(d)/r0)^(r1-2) * K^2 * dK/(2*K) )
            //     = ( -r1/(2.r0) * (sqrt(d)/r0)^(r1-2) * K^2 * (r1-2)/(2*d) ) + ( -r1/(2.r0) * (sqrt(d)/r0)^(r1-2) * K^2 * dK/(2*K) )
            //     = ( dK*(r1-2)/(2*d) ) + ( dK^2/(2*K) )
            //     = ( dK*(r1-2)/(2*d) ) + ( d*dK*dK/(2*d*K) )
            //     = ( (r1-2) + d*dK/K )*dK/(2*d)

            T tmp(diffis);

            OP_sqrt(tmp);
            scaldiv(tmp,r(0));

            res  = pow(tmp,(T) r(1));
            res += 1.0;
            OP_einv(res);

            T tmpb;
            tmpb = pow(tmp,(T) r(1)-2.0);

            diffgrad  = res;
            diffgrad *= res;
            diffgrad *= -0.5;
            scaldiv(diffgrad,r(0));
            scalmul(diffgrad,r(1));
            diffgrad *= tmpb;

            diffdiffgrad  = diffgrad;
            diffdiffgrad /= res;
            diffdiffgrad *= diffis;
            scaladd(diffdiffgrad,r(1));
            diffdiffgrad -= 2.0;
            diffdiffgrad *= diffgrad;
            diffdiffgrad *= 0.5;
            diffdiffgrad /= diffis;

            break;
        }

        case 24:
        {
            // K = (1-(z^i0))/(1-z)
            // dK = ( -i0.(z^(i0-1)) + (1-(z^i0))/(1-z) )/(1-z) 
            //    = ( -i0.(z^i0)/z + K )/(1-z)                         if ( i0 >= 1 )
            // d2K = ( -i0.(i0-1).(z^(i0))/z^2 + dK )/(1-z) + dK/(1-z)
            //     = ( -i0.(i0-1).(z^(i0))/z^2 + 2*K )/(1-z)           if ( i0 >= 2 )
            //
            // Ill-defined at z = 1
            //
            // z -> z/(r0.r0) and blah blah

            T zsc(xyprod);

            scaldiv(zsc,r(0));
            scaldiv(zsc,r(0));

            if ( ( (double) abs2(zsc) > 1e-12 ) || ( ic(0) == 0 ) )
            {
                T tmp = zsc;

                raiseto(tmp,ic(0));

                res  = 1.0;
                res -= tmp;
                res /= (1.0-zsc);
            }

            if ( ( (double) abs2(zsc) > 1e-12 ) || ( ic(0) == 1 ) )
            {
                T tmp = zsc;

                raiseto(tmp,ic(0)-1);

                xygrad  = tmp;
                xygrad *= -ic(0);
                xygrad += res;
                xygrad /= (1.0-zsc);
                scaldiv(xygrad,r(0));
                scaldiv(xygrad,r(0));
            }

            if ( ( (double) abs2(zsc) > 1e-12 ) || ( ic(0) == 2 ) )
            {
                T tmp = xyprod;

                raiseto(tmp,ic(0)-2);

                xyxygrad  = tmp;
                xyxygrad *= -ic(0)*(ic(0)-1);
                xyxygrad += res;
                xyxygrad += res;
                xyxygrad /= (1.0-zsc);
                scaldiv(xyxygrad,r(0));
                scaldiv(xyxygrad,r(0));
                scaldiv(xyxygrad,r(0));
                scaldiv(xyxygrad,r(0));
            }

            break;
        }

        case 25:
        {
            // K = pi.cosh(pi-(sqrt(d)/r0))
            // dK = -pi/(2.r0) * sinh(pi-sqrt(d)/r0) / sqrt(d)
            // d2K = pi^2/(4.r0*r0) * cosh(pi-sqrt(d)/r0) / d  - 1/2d dK
            //     = pi/(4.r0*r0) * K / d  - 1/2d dK
            //     = ( pi*K/(2.r0*r0) - dK )/(2*d)

            T tmp(diffis);

            OP_sqrt(tmp);

            res  = tmp;
            scaldiv(res,r(0));
            res -= NUMBASE_PI;
            res *= -1.0;
            OP_cosh(res);
            res *= NUMBASE_PI;

            diffgrad  = tmp;
            scaldiv(diffgrad,r(0));
            diffgrad -= NUMBASE_PI;
            diffgrad *= -1.0;
            OP_sinh(diffgrad);
            diffgrad *= NUMBASE_PION2;            
            diffgrad *= -0.5;
            scaldiv(diffgrad,r(0));
            diffgrad /= tmp;

            diffdiffgrad  = res;
            diffdiffgrad *= NUMBASE_PI;
            diffdiffgrad *= 0.5;
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));
            diffdiffgrad -= diffgrad;
            diffdiffgrad *= 0.5;
            diffdiffgrad /= diffis;

            break;
        }

        case 26:
        {
            // K = ((d/r0)^(r1+0.5))
            // dK = 1/r0 * (r1+0.5) * (d/r0)^(r1-0.5)
            //    = 1/r0 * (r1+0.5) * K/d
            // d2K = 1/r0^2 * (r1+0.5) * (r1-0.5) * (d/r0)^(r1-1.5)
            //     = 1/r0 * (r1-0.5) * dK/d

            res  = diffis;
            scaldiv(res,r(0));
            res  = pow(res,((T) r(1))+0.5);

            if ( (double) abs2(diffis) > 1e-12 )
            {
                diffgrad = 0.5;
                scaladd(diffgrad,r(1));
                scaldiv(diffgrad,r(0));
                diffgrad *= res;
                diffgrad /= diffis;

                diffdiffgrad = -0.5;
                scaladd(diffdiffgrad,r(1));
                scaldiv(diffdiffgrad,r(0));
                diffdiffgrad *= diffgrad;
                diffdiffgrad /= diffis;
            }

            break;
        }

        case 27:
        {
            // K = ((d/r0)^r1).ln(sqrt(d/r0))
            // dK = (r1/r0).((d/r0)^(r1-1)).ln(sqrt(d/r0)) + ((d/r0)^r1) 1/(sqrt(d/r0)) 1/2 1/sqrt(d*r0)
            //    = (r1/r0).((d/r0)^r1).ln(sqrt(d/r0)).(r0/d) + ((d/r0)^r1)/(2d)
            //    = 2.r1.((d/r0)^r1).ln(sqrt(d/r0))/(2d) + ((d/r0)^r1)/(2d)
            //    = ( 2.r1.ln(sqrt(d/r0)) + 1 ).((d/r0)^r1)/(2d)
            //    = ( ln(sqrt(d/r0)) + 1/(2.r1) ).((d/r0)^r1).(2.r1)/(2d)
            //    = ( ((d/r0)^r1).ln(sqrt(d/r0)) + ((d/r0)^r1)/(2.r1) ).(2.r1)/(2d)
            //    = ( K + ((d/r0)^r1)/(2.r1) ).(2.r1)/(2d)
            // d2K = ( dK + r1/d ((d/r0)^r1)/(2.r1) ).(2.r1)/(2d)  -  ( K + ((d/r0)^r1)/(2.r1) ).(2.r1)/(2d^2) 
            // d2K = ( dK + r1/d ((d/r0)^r1)/(2.r1) ).(2.r1)/(2d)  - 2*( K + ((d/r0)^r1)/(2.r1) ).(2.r1)/((2d)*(2d)) 
            // d2K = ( dK + r1/d ((d/r0)^r1)/(2.r1) ).(2.r1)/(2d)  - (2*dK/(2.r1)).(2.r1)/(2d) 
            // d2K = ( dK + r1/d ((d/r0)^r1)/(2.r1) - 2*dK/(2.r1) ).(2.r1)/(2d) 
            // d2K = ( ( 1 - 1/r1 ).dK + r1/d ((d/r0)^r1)/(2.r1) ).(2.r1)/(2d) 
            // d2K = ( dK + r1*((d/r0)^r1)/(2*(r1-1)*d) )*(2*(r1-1))/(2d) 

            T scalres(diffis);

            scaldiv(scalres,r(0));
            OP_sqrt(scalres);

            T tempres(scalres);

            OP_log(tempres);

            scalres = pow(scalres,(T) r(1));

            res  = scalres;
            res *= tempres;

            if ( (double) abs2(diffis) > 1e-12 )
            {
                diffgrad  = scalres;
                diffgrad *= 0.5;
                scaldiv(diffgrad,r(1));
                diffgrad += res;
                diffgrad *= 2;
                scalmul(diffgrad,r(1));
                diffgrad *= 0.5;
                diffgrad /= diffis;

                diffdiffgrad = scalres;
                scalmul(diffdiffgrad,r(1));
                diffdiffgrad *= 0.5;
                diffdiffgrad /= diffis;
                scaldiv(diffdiffgrad,r(1)-1.0);
                diffdiffgrad += diffgrad;
                scalmul(diffdiffgrad,r(1)-1.0);
                diffdiffgrad /= diffis;
            }

            break;
        }

        case 32:
        {
            // K = r1 if i == j >= 0, 0 otherwise
            // dK = 0.0
            // d2K = 0.0

            if ( ( i == j ) && ( i >= 0 ) )
            {
                res = r(1);
            }

            break;
        }

        case 33:
        {
            // K = 1/(2.r0) ( 1 if real(sqrt(d)) < r0, 0 otherwise )
            // dK = 0
            // d2K = 0

            if ( real(sqrt(diffis)-r(0)) < zerogentype() )
            {
                res = 0.5/r(0);
            }

            break;
        }

        case 34:
        {
            // K = (1-sqrt(d)/r0)/r0 if real(sqrt(d)) < r0, 0 otherwise )
            // dK = 1/r0 1/(2*sqrt(d)*r0)
            //    = 1/r0 1/2 1/r0^2 r0/sqrt(d)
            //    = 1/r0^3 1/2 r0/sqrt(d)
            // d2k = 1/r0 1/2 1/r0^2 -1/2 r0/sqrt(d) 1/d
            //     = -1/r0^3 1/4 r0/sqrt(d) 1/d

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));

            if ( (double) real(tempres-1.0) < 0.0 )
            {
                res  = 1.0;
                res -= tempres;
                scaldiv(res,r(0));

                diffgrad = tempres;
                OP_einv(diffgrad);
                diffgrad /= 2.0;
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));
                scaldiv(diffgrad,r(0));

                diffdiffgrad = tempres;
                OP_einv(diffdiffgrad);
                diffdiffgrad /= diffis;
                diffdiffgrad /= -4.0;
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
                scaldiv(diffdiffgrad,r(0));
            }

            break;
        }

        case 38:
        {
            // K = exp(-sqrt(d)/r0)
            // dK = 1/r0^2 -r0/sqrt(d) exp(-sqrt(d)/r0)
            //    = 1/r0^2 -r0/sqrt(d) K
            // d2K = 1/r0 1/2 1/d 1/sqrt(d) exp(-sqrt(d)/r0) + 1/r0^2 1/d exp(-sqrt(d)/r0)
            //     = 1/r0^4 1/2 r0^2/d r0/sqrt(d) exp(-sqrt(d)/r0) + 1/r0^4 ro^2/d exp(-sqrt(d)/r0)
            //     = 1/r0^4 r0^2/d exp(-sqrt(d)/r0) ( 1/2 r0/sqrt(d) + 1 )
            //     = 1/2 1/r0^2 -r0/sqrt(d) dK ( 2 - r0/sqrt(d) )

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));
            tempres *= -1.0;

            res = tempres;
            OP_exp(res);

            diffgrad  = res;
            diffgrad *= tempres;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));

            diffdiffgrad  = tempres;
            diffdiffgrad += 2.0;
            diffdiffgrad *= diffgrad;
            diffdiffgrad *= tempres;
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));
            diffdiffgrad *= 0.5;

            break;
        }

        case 39:
        {
            // K = (1+((sqrt(3)/r0).sqrt(d))) . exp(-(sqrt(3)/r0).sqrt(d))
            // dK = ( exp( -sqrt(3)/r0 sqrt(d) ) - ( 1 + sqrt(3)/r0 sqrt(d) ) exp( -sqrt(3)/r0 sqrt(d) ) ) sqrt(3)/r0 1/2 1/sqrt(d)
            //    = -((sqrt(3)/r0).sqrt(d)) exp(-((sqrt(3)/r0).sqrt(d))) sqrt(3)/r0 1/2 1/sqrt(d)
            //    = -1/2 3/r0^2 exp(-((sqrt(3)/r0).sqrt(d)))
            //    = -3/2 1/r0^2 exp(-((sqrt(3)/r0).sqrt(d)))
            // d2K = -3/2 1/r0^2 -sqrt(3)/r0 1/sqrt(d) exp(-((sqrt(3)/r0).sqrt(d)))
            //     = (3.sqrt(3))/2 1/r0^3 1/sqrt(d) exp(-((sqrt(3)/r0).sqrt(d)))
            //     = (3.3)/2 1/r0^4 r0/sqrt(3).sqrt(d) exp(-((sqrt(3)/r0).sqrt(d)))
            //     = 3/2 1/r0^4 1/((sqrt(3)/r0).sqrt(d)) exp(-((sqrt(3)/r0).sqrt(d)))
            //     = 1/r0^2 1/((sqrt(3)/r0).sqrt(d)) 3/2 1/r0^2 exp(-((sqrt(3)/r0).sqrt(d)))
            //     = -1/r0^2 1/((sqrt(3)/r0).sqrt(d)) dK

            T tempres(diffis);
            OP_sqrt(tempres);
            scaldiv(tempres,r(0));
            tempres *= sqrt(3.0);

            T expres(tempres);
            expres *= -1.0;
            OP_exp(expres);

            res  = tempres;
            res += 1;
            res *= expres;

            diffgrad  = expres;
            diffgrad *= -1.5;
            scaldiv(diffgrad,r(0));
            scaldiv(diffgrad,r(0));
            
            diffdiffgrad  = diffgrad;
            diffdiffgrad /= tempres;
            scaldiv(diffdiffgrad,r(0));
            scaldiv(diffdiffgrad,r(0));

            break;
        }

        case 42:
        {
            // K = agd(z/(r0.r0))
            // dK = (1/(r0.r0)) sec^2(z/(r0.r0)) / ( 1 - tan^2(z/(r0.r0)) )
            // d2K = (2/(r0.r0.r0.r0)) sec^2(z/(r0.r0)) tan(z/(r0.r0)) / ( 1 - tan^2(z/(r0.r0)) ) + (2/(r0.r0.r0.r0)) sec^4(z/(r0.r0)) tan(z/(r0.r0)) / ( 1 - tan^2(z/(r0.r0)) )^2
            // d2K = (2/(r0.r0.r0.r0)) tan(z/(r0.r0)) sec^2(z/(r0.r0)) / ( 1 - tan^2(z/(r0.r0)) ) + (2/(r0.r0.r0.r0)) tan(z/(r0.r0)) sec^4(z/(r0.r0)) / ( 1 - tan^2(z/(r0.r0)) )^2
            // d2K = (2/(r0.r0)) tan(z/(r0.r0)) dK + (2/(r0.r0)) tan(z/(r0.r0)) dK^2
            // d2K = (2/(r0.r0)) tan(z/(r0.r0)) dK ( 1 + dK )

            T scalres = diffis;
            scaldiv(scalres,r(0));
            scaldiv(scalres,r(0));

            T scz = scalres;
            OP_sec(scz);
            scz *= scz;

            T taz = scalres;
            OP_tan(taz);

            res = scalres;
            OP_agd(res);

            xygrad  = taz;
            xygrad *= taz;
            xygrad *= -1.0;
            xygrad += 1.0;
            OP_einv(xygrad);
            xygrad *= scz;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            xyxygrad  = xygrad;
            xyxygrad += 1.0;
            xyxygrad *= xygrad;
            xyxygrad *= taz;
            xyxygrad *= 2.0;
            scaldiv(xyxygrad,r(0));
            scaldiv(xyxygrad,r(0));

            break;
        }

        case 43:
        {
            // K = log((1+r0.z)/(1-r0.z))
            // dK = r0*(1-r0.z)/(1+r0.z)*( 1/(1-r0.z) + (1+r0.z)/(1-r0.z)^2 )
            //    = r0*(1-r0.z)/(1+r0.z)*( (1-r0.z) + (1+r0.z) )/(1-r0.z)^2
            //    = r0*((1-r0.z)/(1+r0.z))/(1-r0.z)^2
            //    = r0/((1+r0.z)*(1-r0.z))
            // d2K = -r0*r0*( (1-r0.z) - (1+r0.z) )/((1+r0.z)^2*(1-r0.z)^2)
            //     = -2*r0*r0*( r0.z )/((1+r0.z)^2*(1-r0.z)^2)
            //     = -2*r0^3*z/((1+r0.z)^2*(1-r0.z)^2)
            //     = -2*r0*z*dK^2
            // ADDENDUM: r0 -> 1/(r0.r0)

            T tempa = xyprod;

            tempa  = xyprod;
            scaldiv(tempa,r(0));
            scaldiv(tempa,r(0));

            T tempb = tempa;

            tempa += 1.0;
            tempb -= 1.0;

            res  = tempa;
            res /= tempb;
            OP_log(res);

            xygrad  = tempa;
            xygrad *= tempb;
            OP_einv(xygrad);
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            xyxygrad  = -2.0;
            scaldiv(xyxygrad,r(0));
            scaldiv(xyxygrad,r(0));
            xyxygrad  = xyprod;
            xyxygrad  = xygrad;
            xyxygrad  = xygrad;

            break;
        }

        case 44:
        {
            // K = exp(z/(r0.r0))
            // dK = K/(r0.r0)
            // d2K = dK/(r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            scalsub(res,r(1));
            OP_exp(res);

            xygrad = res;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            xyxygrad = xygrad;
            scaldiv(xyxygrad,r(0));
            scaldiv(xyxygrad,r(0));

            break;
        }

        case 45:
        {
            // K = sinh(z/(r0.r0))
            // dK = cosh(z/(r0.r0))/(r0.r0)
            // d2K = K/(r0.r0.r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            OP_sinh(res);

            xygrad = xyprod;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            OP_cosh(xygrad);
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            xyxygrad = res;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 46:
        {
            // K = cosh(z/(r0.r0))
            // dK = sinh(z/(r0.r0))/(r0.r0)
            // d2K = K/(r0.r0.r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            OP_cosh(res);

            xygrad = xyprod;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            OP_sinh(xygrad);
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            xyxygrad = res;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            break;
        }

        case 47:
        {
            // K = sinc(sqrt(d)/r0).cos(2*pi*sqrt(d)/(r0.r1))
            //
            // if d = 0 then dK,d2K = 0

            throw("bugger that");

            break;
        }

        case 100:
        {
            // K = z/(r0.r0)

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            xyxygrad = 0.0;

            break;
        }

        case 103:
        {
            // K = 0 if real(z) < 0, 1 otherwise

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = 0.0;
            }

            else
            {
                res = 1.0;
            }

            xygrad = 0.0;

            xyxygrad = 0.0;

            break;
        }

        case 104:
        {
            // K = 0 if real(z) < 0, z/(r0*r0) otherwise

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = 0.0;

                xygrad = 0.0;
            }

            else
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));

                xygrad = 1.0;
                scaldiv(xygrad,r(0));
                scaldiv(xygrad,r(0));
            }

            xyxygrad = 0.0;

            break;
        }

        case 106:
        {
            // K = r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise

            static T zgt(0.0);

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            if ( xyprod < zgt )
            {
                scalmul(res,r(1));
                scalmul(xygrad,r(1));
            }

            xyxygrad = 0.0;

            break;
        }

        case 200:
        {
            // K = z/(r0.r0) - 1

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));
            res -= 1.0;

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            xyxygrad = 0.0;

            break;
        }

        case 203:
        {
            // K = -1 if real(z) < 0, 1 otherwise

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = -1.0;
            }

            else
            {
                res = 1.0;
            }

            xygrad = 0.0;

            xyxygrad = 0.0;

            break;
        }

        case 204:
        {
            // K = 0 if real(z) < 0, z/(r0*r0) otherwise    - 1

            static T zgt(0.0);

            if ( xyprod < zgt )
            {
                res = -1.0;

                xygrad = 0.0;
            }

            else
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
                res -= 1.0;

                xygrad = 1.0;
                scaldiv(xygrad,r(0));
                scaldiv(xygrad,r(0));
            }

            xyxygrad = 0.0;

            break;
        }

        case 206:
        {
            // K = r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise

            static T zgt(0.0);

            res = xyprod;
            scaldiv(res,r(0));
            scaldiv(res,r(0));

            xygrad = 1.0;
            scaldiv(xygrad,r(0));
            scaldiv(xygrad,r(0));

            if ( xyprod < zgt )
            {
                scalmul(res,r(1));
                scalmul(xygrad,r(1));
            }

            res -= 1.0;

            xyxygrad = 0.0;

            break;
        }

        default:
        {
            throw("fe fi fo fum");

            break;
        }
    }

    res *= (const T &) weight;

    xygrad   *= (const T &) weight;
    diffgrad *= (const T &) weight;

    xyxygrad     *= (const T &) weight;
    diffdiffgrad *= (const T &) weight;

    return;
}



//KERNELSHERE

// gd(0): x'x derivative count
// gd(1): y'y derivative count
// gd(2): <x,y> derivative count
template <class T>
void MercerKernel::dnKKpro(T &res, const Vector<int> &gd, const T &xyprod, const T &diffis, int i, int j, int locindstart, int locindend, int xdim, int m, int isfirstcalc, T &scratch) const
{
    (void) locindend;

    NiceAssert( locindstart == locindend );

    res = 0.0;

    int ii,jj;
    int z = 0;
    int ind = locindstart;
    {
        retVector<gentype> tmpva;

        const Vector<gentype> &r = dRealConstants(ind)(1,1,dRealConstants(ind).size()-1,tmpva);
        const Vector<int> &ic = dIntConstants(ind);

        int ktype = cType(ind);

        if ( ktype == 0 )
        {
            // K = r1
            // dK = 0
            // d2K = 0

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                res = r(1);
            }
        }

        else if ( ktype == 1 )
        {
            // K = z/(r0.r0)
            // dK = 1/(r0.r0)
            // d2K = 0

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
            }

            else if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == 1 ) )
            {
                res = 1.0;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
            }
        }

        else if ( ktype == 2 )
        {
            // K = ( r1 + z/(r0.r0) )^i0
            // dK = i0/(r0.r0) * ( r1 + z/(r0.r0) )^(i0-1)               if ( i0 >= 1 )
            // d2K = i0.(i0-1)/(r0.r0.r0.r0) * ( r1 + z/(r0.r0) )^(i0-2) if ( i0 >= 2 )

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) <= ic(z) ) )
            {
                ii = ic(z);
                res = 1.0;

                for ( jj = 0 ; jj < gd(2) ; jj++ )
                {
                    res *= ii;
                    scaldiv(res,r(0));
                    scaldiv(res,r(0));
                    ii--;
                }

                if ( ii )
                {
                    T temp(xyprod);

                    scaldiv(temp,r(0));
                    scaldiv(temp,r(0));
                    scaladd(temp,r(1));
                    raiseto(temp,ii);

                    res *= temp;
                }
            }
        }

        else if ( ktype == 3 )
        {
            // K = exp(-d/(2.r0.r0))
            // dK = -K/(2*r0*r0)
            // d2K = -dK/(2*r0*r0)

            if ( isfirstcalc )
            {
                scratch  = diffis;
                scratch *= -0.5;
                scaldiv(scratch,r(0));
                scaldiv(scratch,r(0));
                scratch += log(AltDiffNormConst(xdim,m,r(0)));
                scalsub(res,r(1));
                OP_exp(scratch);
            }

            res = scratch;

            int n01 = gd(z)+gd(1);
            int n2  = gd(2);

            for ( jj = 0 ; jj < n01 ; jj++ )
            {
                res *= -0.5;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
            }

            for ( jj = 0 ; jj < n2 ; jj++ )
            {
                scaldiv(res,r(0));
                scaldiv(res,r(0));
            }
        }

        else if ( ktype == 32 )
        {
            // K = r1 if i == j >= 0, 0 otherwise
            // dK = 0.0
            // d2K = 0.0

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) && ( i == j ) && ( i >= 0 ) )
            {
                res = r(1);
            }
        }

        else if ( ktype == 33 )
        {
            // K = 1/(2.r0) ( 1 if real(sqrt(d)) < r0, 0 otherwise )
            // dK = 0
            // d2K = 0

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) && ( real(sqrt(diffis)-r(0)) < zerogentype() ) )
            {
                res = 0.5/r(0);
            }
        }

        else if ( ktype == 44 )
        {
            // K = exp(z/(r0.r0))
            // dK = K/(r0.r0)
            // d2K = dK/(r0.r0)

            if ( isfirstcalc )
            {
                scratch = xyprod;
                scaldiv(scratch,r(0));
                scaldiv(scratch,r(0));
                scalsub(res,r(1));
                OP_exp(scratch);
            }

            if ( ( gd(z) == z ) && ( gd(1) == z ) )
            {
                res = scratch;

                for ( jj = 0 ; jj < gd(2) ; jj++ )
                {
                    scaldiv(res,r(0));
                    scaldiv(res,r(0));
                }
            }
        }

        else if ( ktype == 45 )
        {
            // K = sinh(z/(r0.r0))
            // dK = cosh(z/(r0.r0))/(r0.r0)
            // d2K = K/(r0.r0.r0.r0)

            if ( ( gd(z) == z ) && ( gd(1) == z ) )
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));

                if ( gd(2)%2 )
                {
                    OP_cosh(res);
                }

                else
                {
                    OP_sinh(res);
                }

                for ( jj = 0 ; jj < gd(2) ; jj++ )
                {
                    scaldiv(res,r(0));
                    scaldiv(res,r(0));
                }
            }
        }

        else if ( ktype == 46 )
        {
            // K = cosh(z/(r0.r0))
            // dK = sinh(z/(r0.r0))/(r0.r0)
            // d2K = K/(r0.r0.r0.r0)

            if ( ( gd(z) == z ) && ( gd(1) == z ) )
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));

                if ( gd(2)%2 )
                {
                    OP_sinh(res);
                }

                else
                {
                    OP_cosh(res);
                }

                for ( jj = 0 ; jj < gd(2) ; jj++ )
                {
                    scaldiv(res,r(0));
                    scaldiv(res,r(0));
                }
            }
        }

        else if ( ktype == 100 )
        {
            // K = z/(r0.r0)

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
            }

            else if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == 1 ) )
            {
                res = 1.0;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
            }
        }

        else if ( ktype == 103 )
        {
            // K = 0 if real(z) < 0, 1 otherwise

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                static T zgt(0.0);

                if ( xyprod < zgt )
                {
                    res = 0.0;
                }

                else
                {
                    res = 1.0;
                }
            }
        }

        else if ( ktype == 104 )
        {
            // K = 0 if real(z) < 0, z/(r0*r0) otherwise

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                static T zgt(0.0);

                if ( xyprod < zgt )
                {
                    res = 0.0;
                }

                else
                {
                    res = xyprod;
                    scaldiv(res,r(0));
                    scaldiv(res,r(0));
                }
            }

            else if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == 1 ) )
            {
                static T zgt(0.0);

                if ( xyprod < zgt )
                {
                    res = 0.0;
                }

                else
                {
                    res = 1.0;
                    scaldiv(res,r(0));
                    scaldiv(res,r(0));
                }
            }
        }

        else if ( ktype == 106 )
        {
            // K = r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                static T zgt(0.0);

                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));

                if ( xyprod < zgt )
                {
                    scalmul(res,r(1));
                }
            }

            else if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == 1 ) )
            {
                static T zgt(0.0);

                res = 1.0;
                scaldiv(res,r(0));
                scaldiv(res,r(0));

                if ( xyprod < zgt )
                {
                    scalmul(res,r(1));
                }
            }
        }

        else if ( ktype == 200 )
        {
            // K = z/(r0.r0) - 1

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
                res -= 1.0;
            }

            else if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == 1 ) )
            {
                res = 1.0;
                scaldiv(res,r(0));
                scaldiv(res,r(0));
            }
        }

        else if ( ktype == 203 )
        {
            // K = -1 if real(z) < 0, 1 otherwise

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                static T zgt(0.0);

                if ( xyprod < zgt )
                {
                    res = -1.0;
                }

                else
                {
                    res = 1.0;
                }
            }
        }

        else if ( ktype == 204 )
        {
            // K = 0 if real(z) < 0, z/(r0*r0) otherwise    - 1

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                static T zgt(0.0);

                if ( xyprod < zgt )
                {
                    res = -1.0;
                }

                else
                {
                    res = xyprod;
                    scaldiv(res,r(0));
                    scaldiv(res,r(0));
                    res -= 1.0;
                }
            }

            else if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == 1 ) )
            {
                static T zgt(0.0);

                if ( xyprod < zgt )
                {
                    res = 0.0;
                }

                else
                {
                    res = 1.0;
                    scaldiv(res,r(0));
                    scaldiv(res,r(0));
                }
            }
        }

        else if ( ktype == 206 )
        {
            // K = r1*z/(r0*r0) if real(z) < 0, z/(r0*r0) otherwise

            if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == z ) )
            {
                static T zgt(0.0);

                res = xyprod;
                scaldiv(res,r(0));
                scaldiv(res,r(0));

                if ( xyprod < zgt )
                {
                    scalmul(res,r(1));
                }

                res -= 1.0;
            }

            else if ( ( gd(z) == z ) && ( gd(1) == z ) && ( gd(2) == 1 ) )
            {
                static T zgt(0.0);

                res = 1.0;
                scaldiv(res,r(0));
                scaldiv(res,r(0));

                if ( xyprod < zgt )
                {
                    scalmul(res,r(1));
                }
            }
        }

        else
        {
            throw("fe fi fo fum");
        }

        res *= (const T &) cWeight(locindstart);
    }

    return;
}

















// Pre-process checks

template <class T> 
void MercerKernel::yyydKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const
{
    yyyadKK2(xygrad,xnormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}

template <class T> 
void MercerKernel::yyyd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const
{
    yyyad2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}

template <class T> 
void MercerKernel::yyydnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, const T &bias, const gentype **pxyprod, int i, int j, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const
{
    yyyadnKK2del(sc,n,minmaxind,q,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}

template <class T>
T &MercerKernel::yyyK0(T &res,
                    const T &bias,
                    const gentype **pxyprod,
                    int xdim, int xconsist, int xresmode, int mlid, int assumreal, int justcalcip) const
{
    return yyyaK0(res,bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,justcalcip);
}

template <class T>
T &MercerKernel::yyyK1(T &res,
                    const SparseVector<gentype> &xa, 
                    const vecInfo &xainfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, 
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy00, int assumreal, int justcalcip) const
{
    return yyyaK1(res,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,resmode,mlid,xy00,assumreal,justcalcip);
}

template <class T>
T &MercerKernel::yyyK2(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                    const vecInfo &xainfo, const vecInfo &xbinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib,
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const
{
    return yyyaK2(res,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal,justcalcip);
}

template <class T>
T &MercerKernel::yyyK3(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const
{
    return yyyaK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,justcalcip);
}

template <class T>
T &MercerKernel::yyyK4(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, int id, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const
{
    return yyyaK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,assumreal,justcalcip);
}

template <class T>
T &MercerKernel::yyyKm(int m, T &res,
                    Vector<const SparseVector<gentype> *> &x,
                    Vector<const vecInfo *> &xinfo,
                    const T &bias,
                    Vector<int> &i,
                    const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, 
                    const Matrix<double> *xy, int assumreal, int justcalcip) const
{
    return yyyaKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,assumreal,justcalcip);
}
















// How to treat distributions, diagonal override:

template <class T> 
void MercerKernel::yyyadKK2(T &xygrad, T &xnormgrad, int &minmaxind, 
                          const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                          const vecInfo &xainfo, const vecInfo &xbinfo, 
                          const T &bias, const gentype **pxyprod, 
                          int ia, int ib, 
                          int xdim, int xconsist, int assumreal, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const
{
    int iaset = ( xa.isfarfarfarindpresent(8) && !(xa(8).isValNull()) ) ? 1 : 0;
    int ibset = ( xb.isfarfarfarindpresent(8) && !(xb(8).isValNull()) ) ? 1 : 0;

    int xadiagr  = ( xa.isfarfarfarindpresent(4) && !(xa(4).isValNull()) ) ? 1 : 0;
    int xbdiagr  = ( xb.isfarfarfarindpresent(4) && !(xb(4).isValNull()) ) ? 1 : 0;

    if ( xadiagr || xbdiagr )
    {
        xygrad    = 0.0;
        xnormgrad = 0.0;
    }

    else
    {
        yyybdKK2(xygrad,xnormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);
    }

    return;
}

template <class T> 
void MercerKernel::yyyad2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, 
                           const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                           const vecInfo &xainfo, const vecInfo &xbinfo, 
                           const T &bias, const gentype **pxyprod, 
                           int ia, int ib, 
                           int xdim, int xconsist, int assumreal, int mlid, 
                           const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const
{
    int iaset = ( xa.isfarfarfarindpresent(8) && !(xa(8).isValNull()) ) ? 1 : 0;
    int ibset = ( xb.isfarfarfarindpresent(8) && !(xb(8).isValNull()) ) ? 1 : 0;

    int xadiagr  = ( xa.isfarfarfarindpresent(4) && !(xa(4).isValNull()) ) ? 1 : 0;
    int xbdiagr  = ( xb.isfarfarfarindpresent(4) && !(xb(4).isValNull()) ) ? 1 : 0;

    if ( xadiagr || xbdiagr )
    {
        xygrad         = 0.0;
        xnormgrad      = 0.0;
        xyxygrad       = 0.0;
        xyxnormgrad    = 0.0;
        xyynormgrad    = 0.0;
        xnormxnormgrad = 0.0;
        xnormynormgrad = 0.0;
        ynormynormgrad = 0.0;
    }

    else
    {
        yyybd2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);
    }

    return;
}

template <class T> 
void MercerKernel::yyyadnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, 
                                const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                                const vecInfo &xainfo, const vecInfo &xbinfo, 
                                const T &bias, const gentype **pxyprod, 
                                int ia, int ib, 
                                int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const
{
    int iaset = ( xa.isfarfarfarindpresent(8) && !(xa(8).isValNull()) ) ? 1 : 0;
    int ibset = ( xb.isfarfarfarindpresent(8) && !(xb(8).isValNull()) ) ? 1 : 0;

    int xadiagr  = ( xa.isfarfarfarindpresent(4) && !(xa(4).isValNull()) ) ? 1 : 0;
    int xbdiagr  = ( xb.isfarfarfarindpresent(4) && !(xb(4).isValNull()) ) ? 1 : 0;

    if ( xadiagr || xbdiagr )
    {
        sc.resize(0);
        n.resize(0);
    }

    else
    {
        yyybdnKK2del(sc,n,minmaxind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);
    }

    return;
}

template <class T>
T &MercerKernel::yyyaK0(T &res,
                    const T &bias,
                    const gentype **pxyprod,
                    int xdim, int xconsist, int xresmode, int mlid, int assumreal, int justcalcip) const
{
    return yyybK0(res,bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,justcalcip);
}

template <class T>
T &MercerKernel::yyyaK1(T &res,
                    const SparseVector<gentype> &xa, 
                    const vecInfo &xainfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, 
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy00, int assumreal, int justcalcip) const
{
    int iaset = ( xa.isfarfarfarindpresent(8) && !(xa(8).isValNull()) ) ? 1 : 0;
    int xadiagr = ( xa.isfarfarfarindpresent(4) && !(xa(4).isValNull()) ) ? 1 : 0;
    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );

    int iaupm = xa.nearupsize();

    if ( xadiagr && ( ia >= 0 ) )
    {
        NiceAssert( !justcalcip );

        res = (T) xa.fff(4);
    }

    else if ( xadiagr )
    {
        NiceAssert( !justcalcip );

        res = 0.0;
    }

    else if ( !xafarpresent && !xagradOrder && ( iaupm == 1 ) )
    {
        xKKK1(res,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,assumreal,resmode,mlid,xy00,justcalcip,iaset);
    }

    else
    {
        yyybK1(res,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,resmode,mlid,xy00,iaset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyyaK2(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                    const vecInfo &xainfo, const vecInfo &xbinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib,
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const
{
    int iaset = ( xa.isfarfarfarindpresent(8) && !(xa(8).isValNull()) ) ? 1 : 0;
    int ibset = ( xb.isfarfarfarindpresent(8) && !(xb(8).isValNull()) ) ? 1 : 0;

    int xadiagr = ( xa.isfarfarfarindpresent(4) && !(xa(4).isValNull()) ) ? 1 : 0;
    int xbdiagr = ( xb.isfarfarfarindpresent(4) && !(xb(4).isValNull()) ) ? 1 : 0;

    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xbfarpresent = xb.isfaroffindpresent() ? 1 : 0;

    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xbfarfarpresent = xb.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xbind6present = xb.isfarfarfarindpresent(6) && !(xb(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );
    int xbgradOrder = xbind6present ? ( (int) xb.fff(6) ) : ( xbfarfarpresent ? 1 : 0 );

    int iaupm = xa.nearupsize();
    int ibupm = xb.nearupsize();

    if ( xadiagr && xbdiagr && ( ia == ib ) && ( ia >= 0 ) )
    {
        NiceAssert( !justcalcip );

        res =  (T) xa.fff(4);
        res *= (T) xb.fff(4);
    }

    else if ( xadiagr || xbdiagr )
    {
        NiceAssert( !justcalcip );

        res = 0.0;
    }

    else if ( !xafarpresent && !xbfarpresent && !xagradOrder && !xbgradOrder && ( iaupm == 1 ) && ( ibupm == 1 ) )
    {
        xKKK2(res,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,justcalcip,iaset,ibset);
    }

    else
    {
        yyybK2(res,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,iaset,ibset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyyaK3(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const
{
    int iaset = ( xa.isfarfarfarindpresent(8) && !(xa(8).isValNull()) ) ? 1 : 0;
    int ibset = ( xb.isfarfarfarindpresent(8) && !(xb(8).isValNull()) ) ? 1 : 0;
    int icset = ( xc.isfarfarfarindpresent(8) && !(xc(8).isValNull()) ) ? 1 : 0;

    int xadiagr = ( xa.isfarfarfarindpresent(4) && !(xa(4).isValNull()) ) ? 1 : 0;
    int xbdiagr = ( xb.isfarfarfarindpresent(4) && !(xb(4).isValNull()) ) ? 1 : 0;
    int xcdiagr = ( xc.isfarfarfarindpresent(4) && !(xc(4).isValNull()) ) ? 1 : 0;

    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xbfarpresent = xb.isfaroffindpresent() ? 1 : 0;
    int xcfarpresent = xc.isfaroffindpresent() ? 1 : 0;

    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xbfarfarpresent = xb.isfarfaroffindpresent() ? 1 : 0;
    int xcfarfarpresent = xc.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xbind6present = xb.isfarfarfarindpresent(6) && !(xb(6).isValNull());
    int xcind6present = xc.isfarfarfarindpresent(6) && !(xc(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );
    int xbgradOrder = xbind6present ? ( (int) xb.fff(6) ) : ( xbfarfarpresent ? 1 : 0 );
    int xcgradOrder = xcind6present ? ( (int) xc.fff(6) ) : ( xcfarfarpresent ? 1 : 0 );

    int iaupm = xa.nearupsize();
    int ibupm = xb.nearupsize();
    int icupm = xc.nearupsize();

    if ( xadiagr && xbdiagr && xcdiagr && ( ia == ib ) && ( ia == ic ) && ( ia >= 0 ) )
    {
        NiceAssert( !justcalcip );

        res =  (T) xa.fff(4);
        res *= (T) xb.fff(4);
        res *= (T) xc.fff(4);
    }

    else if ( xadiagr || xbdiagr || xcdiagr )
    {
        NiceAssert( !justcalcip );

        res = 0.0;
    }

    else if ( !xafarpresent && !xbfarpresent && !xcfarpresent && !xagradOrder && !xbgradOrder && !xcgradOrder && ( iaupm == 1 ) && ( ibupm == 1 ) && ( icupm == 1 ) )
    {
        xKKK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,assumreal,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,justcalcip,iaset,ibset,icset);
    }

    else
    {
        yyybK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,iaset,ibset,icset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyyaK4(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, int id, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const
{
    int iaset = ( xa.isfarfarfarindpresent(8) && !(xa(8).isValNull()) ) ? 1 : 0;
    int ibset = ( xb.isfarfarfarindpresent(8) && !(xb(8).isValNull()) ) ? 1 : 0;
    int icset = ( xc.isfarfarfarindpresent(8) && !(xc(8).isValNull()) ) ? 1 : 0;
    int idset = ( xd.isfarfarfarindpresent(8) && !(xd(8).isValNull()) ) ? 1 : 0;

    int xadiagr = ( xa.isfarfarfarindpresent(4) && !(xa(4).isValNull()) ) ? 1 : 0;
    int xbdiagr = ( xb.isfarfarfarindpresent(4) && !(xb(4).isValNull()) ) ? 1 : 0;
    int xcdiagr = ( xc.isfarfarfarindpresent(4) && !(xc(4).isValNull()) ) ? 1 : 0;
    int xddiagr = ( xd.isfarfarfarindpresent(4) && !(xd(4).isValNull()) ) ? 1 : 0;

    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xbfarpresent = xb.isfaroffindpresent() ? 1 : 0;
    int xcfarpresent = xc.isfaroffindpresent() ? 1 : 0;
    int xdfarpresent = xd.isfaroffindpresent() ? 1 : 0;

    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xbfarfarpresent = xb.isfarfaroffindpresent() ? 1 : 0;
    int xcfarfarpresent = xc.isfarfaroffindpresent() ? 1 : 0;
    int xdfarfarpresent = xd.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xbind6present = xb.isfarfarfarindpresent(6) && !(xb(6).isValNull());
    int xcind6present = xc.isfarfarfarindpresent(6) && !(xc(6).isValNull());
    int xdind6present = xd.isfarfarfarindpresent(6) && !(xd(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );
    int xbgradOrder = xbind6present ? ( (int) xb.fff(6) ) : ( xbfarfarpresent ? 1 : 0 );
    int xcgradOrder = xcind6present ? ( (int) xc.fff(6) ) : ( xcfarfarpresent ? 1 : 0 );
    int xdgradOrder = xdind6present ? ( (int) xd.fff(6) ) : ( xdfarfarpresent ? 1 : 0 );

    int iaupm = xa.nearupsize();
    int ibupm = xb.nearupsize();
    int icupm = xc.nearupsize();
    int idupm = xd.nearupsize();

    if ( xadiagr && xbdiagr && xcdiagr && xddiagr && ( ia == ib ) && ( ia == ic ) && ( ia == id ) && ( ia >= 0 ) )
    {
        NiceAssert( !justcalcip );

        res =  (T) xa.fff(4);
        res *= (T) xb.fff(4);
        res *= (T) xc.fff(4);
        res *= (T) xd.fff(4);
    }

    else if ( xadiagr || xbdiagr || xcdiagr || xddiagr )
    {
        NiceAssert( !justcalcip );

        res = 0.0;
    }

    else if ( !xafarpresent && !xbfarpresent && !xcfarpresent && !xcfarpresent && !xdfarpresent && !xagradOrder && !xbgradOrder && !xcgradOrder && !xdgradOrder && ( iaupm == 1 ) && ( ibupm == 1 ) && ( icupm == 1 ) && ( idupm == 1 ) )
    {
        xKKK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,assumreal,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,justcalcip,iaset,ibset,icset,idset);
    }

    else
    {
        yyybK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyyaKm(int m, T &res,
                    Vector<const SparseVector<gentype> *> &x,
                    Vector<const vecInfo *> &xinfo,
                    const T &bias,
                    Vector<int> &i,
                    const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, 
                    const Matrix<double> *xy, int assumreal, int justcalcip) const
{
    Vector<int> iiset(x.size());
    Vector<int> xdiagr(x.size());

    if ( x.size() )
    {
        int ii;

        for ( ii = 0 ; ii < x.size() ; ii++ )
        {
            iiset("&",ii)  = ( (*(x(ii))).isfarfarfarindpresent(8) && !((*(x(ii)))(8).isValNull()) ) ? 1 : 0;
            xdiagr("&",ii) = ( (*(x(ii))).isfarfarfarindpresent(4) && !((*(x(ii)))(4).isValNull()) ) ? 1 : 0;
        }

        if ( xdiagr != zeroint() )
        {
            if ( ( xdiagr == xdiagr(zeroint()) ) && ( i == i(zeroint()) ) && ( i(zeroint()) >= 0 ) )
            {
                NiceAssert( !justcalcip );

                res = 0.0;

                for ( ii = 0 ; ii < x.size() ; ii++ )
                {
                    if ( !ii )
                    {
                        res = (T) (*(x(ii))).fff(4);
                    }

                    else
                    {
                        res *= (T) (*(x(ii))).fff(4);
                    }
                }
            }

            else
            {
                NiceAssert( !justcalcip );

                res = 0.0;
            }
        }

        else
        {
            yyybKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,&iiset,assumreal,justcalcip);
        }
    }

    else
    {
        yyybKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,&iiset,assumreal,justcalcip);
    }

    return res;
}
















// Pre-process Rank
//
// NB: nearref actually just returns the vector itself, so gradient parts are passed through by this

template <class T> 
void MercerKernel::yyybdKK2(T &xygrad, T &xnormgrad, int &minmaxind, 
                            const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                            const vecInfo &xainfo, const vecInfo &xbinfo, 
                            const T &bias, const gentype **pxyprod, 
                            int ia, int ib, 
                            int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset) const
{
#ifndef NDEBUG
    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xbfarpresent = xb.isfaroffindpresent() ? 1 : 0;

    NiceAssert( !xafarpresent );
    NiceAssert( !xbfarpresent );
#endif

    yyycdKK2(xygrad,xnormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    return;
}

template <class T> 
void MercerKernel::yyybd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, 
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                             const vecInfo &xainfo, const vecInfo &xbinfo, 
                             const T &bias, const gentype **pxyprod, 
                             int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset) const
{
#ifndef NDEBUG
    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xbfarpresent = xb.isfaroffindpresent() ? 1 : 0;

    NiceAssert( !xafarpresent );
    NiceAssert( !xbfarpresent );
#endif

    yyycd2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    return;
}

template <class T> 
void MercerKernel::yyybdnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, 
                               const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                               const vecInfo &xainfo, const vecInfo &xbinfo, 
                               const T &bias, const gentype **pxyprod, 
                               int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset) const
{
#ifndef NDEBUG
    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xbfarpresent = xb.isfaroffindpresent() ? 1 : 0;

    NiceAssert( !xafarpresent );
    NiceAssert( !xbfarpresent );
#endif

    yyycdnKK2del(sc,n,minmaxind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    return;
}



template <class T>
T &MercerKernel::yyybK0(T &res,
                    const T &bias,
                    const gentype **pxyprod,
                    int xdim, int xconsist, int xresmode, int mlid, int assumreal, int justcalcip) const
{
    return yyycK0(res,bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,justcalcip);
}

template <class T>
T &MercerKernel::yyybK1(T &res,
                    const SparseVector<gentype> &xa, 
                    const vecInfo &xainfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, 
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy00, int iaset, int assumreal, int justcalcip) const
{
    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;

    NiceAssert( !( justcalcip && xafarpresent ) );

    if ( xafarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        int iia = -(((ia+1)*100)+1);

        T tmp;

               yyycK1(res,xanear,xanearinfo,bias,NULL,ia ,xdim,xconsist,resmode,mlid,NULL,iaset,assumreal,justcalcip);
        res -= yyycK1(tmp,xafar ,xafarinfo ,bias,NULL,iia,xdim,xconsist,resmode,mlid,NULL,iaset,assumreal,justcalcip);
    }

    else
    {
        yyycK1(res,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,resmode,mlid,xy00,iaset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyybK2(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                    const vecInfo &xainfo, const vecInfo &xbinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib,
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, int iaset, int ibset, int assumreal, int justcalcip) const
{
    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xbfarpresent = xb.isfaroffindpresent() ? 1 : 0;

    NiceAssert( !( justcalcip && ( xafarpresent || xbfarpresent ) ) );

    if ( xafarpresent && xbfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iib = -(((ib+1)*100)+1);

        T tmp;

               yyycK2(res,xanear,xbnear,xanearinfo,xbnearinfo,bias,NULL,ia ,ib ,xdim,xconsist,resmode,mlid,NULL,NULL,NULL,iaset,ibset,assumreal,justcalcip);
        res -= yyycK2(tmp,xanear,xbfar ,xanearinfo,xbfarinfo ,bias,NULL,ia ,iib,xdim,xconsist,resmode,mlid,NULL,NULL,NULL,iaset,ibset,assumreal,justcalcip);
        res -= yyycK2(tmp,xafar ,xbnear,xafarinfo ,xbnearinfo,bias,NULL,iia,ib ,xdim,xconsist,resmode,mlid,NULL,NULL,NULL,iaset,ibset,assumreal,justcalcip);
        res += yyycK2(tmp,xafar ,xbfar ,xafarinfo ,xbfarinfo ,bias,NULL,iia,iib,xdim,xconsist,resmode,mlid,NULL,NULL,NULL,iaset,ibset,assumreal,justcalcip);
    }

    else if ( xafarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        int iia = -(((ia+1)*100)+1);

        T tmp;

               yyycK2(res,xanear,xb,xanearinfo,xbinfo,bias,NULL,ia ,ib ,xdim,xconsist,resmode,mlid,NULL,NULL,NULL,iaset,ibset,assumreal,justcalcip);
        res -= yyycK2(tmp,xafar ,xb,xafarinfo ,xbinfo,bias,NULL,iia,ib ,xdim,xconsist,resmode,mlid,NULL,NULL,NULL,iaset,ibset,assumreal,justcalcip);
    }

    else if ( xbfarpresent )
    {
        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        int iib = -(((ib+1)*100)+1);

        T tmp;

               yyycK2(res,xa,xbnear,xainfo,xbnearinfo,bias,NULL,ia ,ib ,xdim,xconsist,resmode,mlid,NULL,NULL,NULL,iaset,ibset,assumreal,justcalcip);
        res -= yyycK2(tmp,xa,xbfar ,xainfo,xbfarinfo ,bias,NULL,ia ,iib,xdim,xconsist,resmode,mlid,NULL,NULL,NULL,iaset,ibset,assumreal,justcalcip);
    }

    else
    {
        yyycK2(res,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,iaset,ibset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyybK3(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iaset, int ibset, int icset, int assumreal, int justcalcip) const
{
    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xbfarpresent = xb.isfaroffindpresent() ? 1 : 0;
    int xcfarpresent = xc.isfaroffindpresent() ? 1 : 0;

    NiceAssert( !( justcalcip && ( xafarpresent || xbfarpresent || xcfarpresent ) ) );

    if ( xafarpresent && xbfarpresent && xcfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iib = -(((ib+1)*100)+1);
        int iic = -(((ic+1)*100)+1);

        T tmp;

               yyycK3(res,xanear,xbnear,xcnear,xanearinfo,xbnearinfo,xcnearinfo,bias,NULL,ia ,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xanear,xbnear,xcfar ,xanearinfo,xbnearinfo,xcfarinfo ,bias,NULL,ia ,ib ,iic,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xanear,xbfar ,xcnear,xanearinfo,xbfarinfo ,xcnearinfo,bias,NULL,ia ,iib,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res += yyycK3(tmp,xanear,xbfar ,xcfar ,xanearinfo,xbfarinfo ,xcfarinfo ,bias,NULL,ia ,iib,iic,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xafar ,xbnear,xcnear,xafarinfo ,xbnearinfo,xcnearinfo,bias,NULL,iia,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res += yyycK3(tmp,xafar ,xbnear,xcfar ,xafarinfo ,xbnearinfo,xcfarinfo ,bias,NULL,iia,ib ,iic,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res += yyycK3(tmp,xafar ,xbfar ,xcnear,xafarinfo ,xbfarinfo ,xcnearinfo,bias,NULL,iia,iib,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xafar ,xbfar ,xcfar ,xafarinfo ,xbfarinfo ,xcfarinfo ,bias,NULL,iia,iib,iic,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
    }

    else if ( xafarpresent && xbfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iib = -(((ib+1)*100)+1);

        T tmp;

               yyycK3(res,xanear,xbnear,xc,xanearinfo,xbnearinfo,xcinfo,bias,NULL,ia ,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xanear,xbfar ,xc,xanearinfo,xbfarinfo ,xcinfo,bias,NULL,ia ,iib,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xafar ,xbnear,xc,xafarinfo ,xbnearinfo,xcinfo,bias,NULL,iia,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res += yyycK3(tmp,xafar ,xbfar ,xc,xafarinfo ,xbfarinfo ,xcinfo,bias,NULL,iia,iib,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
    }

    else if ( xafarpresent && xcfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iic = -(((ic+1)*100)+1);

        T tmp;

               yyycK3(res,xanear,xb,xcnear,xanearinfo,xbinfo,xcnearinfo,bias,NULL,ia ,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xanear,xb,xcfar ,xanearinfo,xbinfo,xcfarinfo ,bias,NULL,ia ,ib ,iic,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xafar ,xb,xcnear,xafarinfo ,xbinfo,xcnearinfo,bias,NULL,iia,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res += yyycK3(tmp,xafar ,xb,xcfar ,xafarinfo ,xbinfo,xcfarinfo ,bias,NULL,iia,ib ,iic,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
    }

    else if ( xbfarpresent && xcfarpresent )
    {
        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        int iib = -(((ib+1)*100)+1);
        int iic = -(((ic+1)*100)+1);

        T tmp;

               yyycK3(res,xa,xbnear,xcnear,xainfo,xbnearinfo,xcnearinfo,bias,NULL,ia ,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xa,xbnear,xcfar ,xainfo,xbnearinfo,xcfarinfo ,bias,NULL,ia ,ib ,iic,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xa,xbfar ,xcnear,xainfo,xbfarinfo ,xcnearinfo,bias,NULL,ia ,iib,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res += yyycK3(tmp,xa,xbfar ,xcfar ,xainfo,xbfarinfo ,xcfarinfo ,bias,NULL,ia ,iib,iic,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
    }

    else if ( xafarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        int iia = -(((ia+1)*100)+1);

        T tmp;

               yyycK3(res,xanear,xb,xc,xanearinfo,xbinfo,xcinfo,bias,NULL,ia ,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xafar ,xb,xc,xafarinfo ,xbinfo,xcinfo,bias,NULL,iia,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
    }

    else if ( xbfarpresent )
    {
        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        int iib = -(((ib+1)*100)+1);

        T tmp;

               yyycK3(res,xa,xbnear,xc,xainfo,xbnearinfo,xcinfo,bias,NULL,ia ,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xa,xbfar ,xc,xainfo,xbfarinfo ,xcinfo,bias,NULL,ia ,iib,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
    }

    else if ( xcfarpresent )
    {
        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        int iic = -(((ic+1)*100)+1);

        T tmp;

               yyycK3(res,xa,xb,xcnear,xainfo,xbinfo,xcnearinfo,bias,NULL,ia ,ib ,ic ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
        res -= yyycK3(tmp,xa,xb,xcfar ,xainfo,xbinfo,xcfarinfo ,bias,NULL,ia ,ib ,iic,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,assumreal,justcalcip);
    }

    else
    {
        yyycK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,iaset,ibset,icset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyybK4(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, int id, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int assumreal, int justcalcip) const
{
    int xafarpresent = xa.isfaroffindpresent() ? 1 : 0;
    int xbfarpresent = xb.isfaroffindpresent() ? 1 : 0;
    int xcfarpresent = xc.isfaroffindpresent() ? 1 : 0;
    int xdfarpresent = xd.isfaroffindpresent() ? 1 : 0;

    NiceAssert( !( justcalcip && ( xafarpresent || xbfarpresent || xcfarpresent || xdfarpresent ) ) );

    if ( xafarpresent && xbfarpresent && xcfarpresent && xdfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const SparseVector<gentype> &xdnear = xd.nearref();
        const SparseVector<gentype> &xdfar  = xd.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        const vecInfo &xdnearinfo = xdinfo(0,-1);
        const vecInfo &xdfarinfo  = xdinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iib = -(((ib+1)*100)+1);
        int iic = -(((ic+1)*100)+1);
        int iid = -(((id+1)*100)+1);

        T tmp;

               yyycK4(res,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xbnear,xcnear,xdfar ,xanearinfo,xbnearinfo,xcnearinfo,xdfarinfo ,bias,NULL,ia, ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xbnear,xcfar ,xdnear,xanearinfo,xbnearinfo,xcfarinfo ,xdnearinfo,bias,NULL,ia, ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xanear,xbnear,xcfar ,xdfar ,xanearinfo,xbnearinfo,xcfarinfo ,xdfarinfo ,bias,NULL,ia, ib, iic,iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xbfar ,xcnear,xdnear,xanearinfo,xbfarinfo ,xcnearinfo,xdnearinfo,bias,NULL,ia, iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xanear,xbfar ,xcnear,xdfar ,xanearinfo,xbfarinfo ,xcnearinfo,xdfarinfo ,bias,NULL,ia, iib,ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xanear,xbfar ,xcfar ,xdnear,xanearinfo,xbfarinfo ,xcfarinfo ,xdnearinfo,bias,NULL,ia, iib,iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xbfar ,xcfar ,xdfar ,xanearinfo,xbfarinfo ,xcfarinfo ,xdfarinfo ,bias,NULL,ia, iib,iic,iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xbnear,xcnear,xdnear,xafarinfo ,xbnearinfo,xcnearinfo,xdnearinfo,bias,NULL,iia,ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xbnear,xcnear,xdfar ,xafarinfo ,xbnearinfo,xcnearinfo,xdfarinfo ,bias,NULL,iia,ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xbnear,xcfar ,xdnear,xafarinfo ,xbnearinfo,xcfarinfo ,xdnearinfo,bias,NULL,iia,ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xbnear,xcfar ,xdfar ,xafarinfo ,xbnearinfo,xcfarinfo ,xdfarinfo ,bias,NULL,iia,ib, iic,iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xbfar ,xcnear,xdnear,xafarinfo ,xbfarinfo ,xcnearinfo,xdnearinfo,bias,NULL,iia,iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xbfar ,xcnear,xdfar ,xafarinfo ,xbfarinfo ,xcnearinfo,xdfarinfo ,bias,NULL,iia,iib,ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xbfar ,xcfar ,xdnear,xafarinfo ,xbfarinfo ,xcfarinfo ,xdnearinfo,bias,NULL,iia,iib,iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xbfar ,xcfar ,xdfar ,xafarinfo ,xbfarinfo ,xcfarinfo ,xdfarinfo ,bias,NULL,iia,iib,iic,iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xafarpresent && xbfarpresent && xcfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iib = -(((ib+1)*100)+1);
        int iic = -(((ic+1)*100)+1);

        T tmp;

               yyycK4(res,xanear,xbnear,xcnear,xd,xanearinfo,xbnearinfo,xcnearinfo,xdinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xbnear,xcfar ,xd,xanearinfo,xbnearinfo,xcfarinfo ,xdinfo,bias,NULL,ia, ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xbfar ,xcnear,xd,xanearinfo,xbfarinfo ,xcnearinfo,xdinfo,bias,NULL,ia, iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xanear,xbfar ,xcfar ,xd,xanearinfo,xbfarinfo ,xcfarinfo ,xdinfo,bias,NULL,ia, iib,iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xbnear,xcnear,xd,xafarinfo ,xbnearinfo,xcnearinfo,xdinfo,bias,NULL,iia,ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xbnear,xcfar ,xd,xafarinfo ,xbnearinfo,xcfarinfo ,xdinfo,bias,NULL,iia,ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xbfar ,xcnear,xd,xafarinfo ,xbfarinfo ,xcnearinfo,xdinfo,bias,NULL,iia,iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xbfar ,xcfar ,xd,xafarinfo ,xbfarinfo ,xcfarinfo ,xdinfo,bias,NULL,iia,iib,iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xafarpresent && xbfarpresent && xdfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const SparseVector<gentype> &xdnear = xd.nearref();
        const SparseVector<gentype> &xdfar  = xd.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        const vecInfo &xdnearinfo = xdinfo(0,-1);
        const vecInfo &xdfarinfo  = xdinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iib = -(((ib+1)*100)+1);
        int iid = -(((id+1)*100)+1);

        T tmp;

               yyycK4(res,xanear,xbnear,xc,xdnear,xanearinfo,xbnearinfo,xcinfo,xdnearinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xbnear,xc,xdfar ,xanearinfo,xbnearinfo,xcinfo,xdfarinfo ,bias,NULL,ia, ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xbfar ,xc,xdnear,xanearinfo,xbfarinfo ,xcinfo,xdnearinfo,bias,NULL,ia, iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xanear,xbfar ,xc,xdfar ,xanearinfo,xbfarinfo ,xcinfo,xdfarinfo ,bias,NULL,ia, iib,ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xbnear,xc,xdnear,xafarinfo ,xbnearinfo,xcinfo,xdnearinfo,bias,NULL,iia,ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xbnear,xc,xdfar ,xafarinfo ,xbnearinfo,xcinfo,xdfarinfo ,bias,NULL,iia,ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xbfar ,xc,xdnear,xafarinfo ,xbfarinfo ,xcinfo,xdnearinfo,bias,NULL,iia,iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xbfar ,xc,xdfar ,xafarinfo ,xbfarinfo ,xcinfo,xdfarinfo ,bias,NULL,iia,iib,ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xafarpresent && xcfarpresent && xdfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const SparseVector<gentype> &xdnear = xd.nearref();
        const SparseVector<gentype> &xdfar  = xd.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        const vecInfo &xdnearinfo = xdinfo(0,-1);
        const vecInfo &xdfarinfo  = xdinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iic = -(((ic+1)*100)+1);
        int iid = -(((id+1)*100)+1);

        T tmp;

               yyycK4(res,xanear,xb,xcnear,xdnear,xanearinfo,xbinfo,xcnearinfo,xdnearinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xb,xcnear,xdfar ,xanearinfo,xbinfo,xcnearinfo,xdfarinfo ,bias,NULL,ia, ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xb,xcfar ,xdnear,xanearinfo,xbinfo,xcfarinfo ,xdnearinfo,bias,NULL,ia, ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xanear,xb,xcfar ,xdfar ,xanearinfo,xbinfo,xcfarinfo ,xdfarinfo ,bias,NULL,ia, ib, iic,iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xb,xcnear,xdnear,xafarinfo ,xbinfo,xcnearinfo,xdnearinfo,bias,NULL,iia,ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xb,xcnear,xdfar ,xafarinfo ,xbinfo,xcnearinfo,xdfarinfo ,bias,NULL,iia,ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xb,xcfar ,xdnear,xafarinfo ,xbinfo,xcfarinfo ,xdnearinfo,bias,NULL,iia,ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xb,xcfar ,xdfar ,xafarinfo ,xbinfo,xcfarinfo ,xdfarinfo ,bias,NULL,iia,ib, iic,iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xbfarpresent && xcfarpresent && xdfarpresent )
    {
        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const SparseVector<gentype> &xdnear = xd.nearref();
        const SparseVector<gentype> &xdfar  = xd.farref();

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        const vecInfo &xdnearinfo = xdinfo(0,-1);
        const vecInfo &xdfarinfo  = xdinfo(1,-1);

        int iib = -(((ib+1)*100)+1);
        int iic = -(((ic+1)*100)+1);
        int iid = -(((id+1)*100)+1);

        T tmp;

               yyycK4(res,xa,xbnear,xcnear,xdnear,xainfo,xbnearinfo,xcnearinfo,xdnearinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xbnear,xcnear,xdfar ,xainfo,xbnearinfo,xcnearinfo,xdfarinfo ,bias,NULL,ia, ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xbnear,xcfar ,xdnear,xainfo,xbnearinfo,xcfarinfo ,xdnearinfo,bias,NULL,ia, ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xa,xbnear,xcfar ,xdfar ,xainfo,xbnearinfo,xcfarinfo ,xdfarinfo ,bias,NULL,ia, ib, iic,iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xbfar ,xcnear,xdnear,xainfo,xbfarinfo ,xcnearinfo,xdnearinfo,bias,NULL,ia, iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xa,xbfar ,xcnear,xdfar ,xainfo,xbfarinfo ,xcnearinfo,xdfarinfo ,bias,NULL,ia, iib,ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xa,xbfar ,xcfar ,xdnear,xainfo,xbfarinfo ,xcfarinfo ,xdnearinfo,bias,NULL,ia, iib,iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xbfar ,xcfar ,xdfar ,xainfo,xbfarinfo ,xcfarinfo ,xdfarinfo ,bias,NULL,ia, iib,iic,iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xafarpresent && xbfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iib = -(((ib+1)*100)+1);

        T tmp;

               yyycK4(res,xanear,xbnear,xc,xd,xanearinfo,xbnearinfo,xcinfo,xdinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xbfar ,xc,xd,xanearinfo,xbfarinfo ,xcinfo,xdinfo,bias,NULL,ia, iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xbnear,xc,xd,xafarinfo ,xbnearinfo,xcinfo,xdinfo,bias,NULL,iia,ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xbfar ,xc,xd,xafarinfo ,xbfarinfo ,xcinfo,xdinfo,bias,NULL,iia,iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xafarpresent && xcfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iic = -(((ic+1)*100)+1);

        T tmp;

               yyycK4(res,xanear,xb,xcnear,xd,xanearinfo,xbinfo,xcnearinfo,xdinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xb,xcfar ,xd,xanearinfo,xbinfo,xcfarinfo ,xdinfo,bias,NULL,ia, ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xb,xcnear,xd,xafarinfo ,xbinfo,xcnearinfo,xdinfo,bias,NULL,iia,ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xb,xcfar ,xd,xafarinfo ,xbinfo,xcfarinfo ,xdinfo,bias,NULL,iia,ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xafarpresent && xdfarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const SparseVector<gentype> &xdnear = xd.nearref();
        const SparseVector<gentype> &xdfar  = xd.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        const vecInfo &xdnearinfo = xdinfo(0,-1);
        const vecInfo &xdfarinfo  = xdinfo(1,-1);

        int iia = -(((ia+1)*100)+1);
        int iid = -(((id+1)*100)+1);

        T tmp;

               yyycK4(res,xanear,xb,xc,xdnear,xanearinfo,xbinfo,xcinfo,xdnearinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xanear,xb,xc,xdfar ,xanearinfo,xbinfo,xcinfo,xdfarinfo ,bias,NULL,ia, ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xb,xc,xdnear,xafarinfo ,xbinfo,xcinfo,xdnearinfo,bias,NULL,iia,ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xafar ,xb,xc,xdfar ,xafarinfo ,xbinfo,xcinfo,xdfarinfo ,bias,NULL,iia,ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xbfarpresent && xcfarpresent )
    {
        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        int iib = -(((ib+1)*100)+1);
        int iic = -(((ic+1)*100)+1);

        T tmp;

               yyycK4(res,xa,xbnear,xcnear,xd,xainfo,xbnearinfo,xcnearinfo,xdinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xbnear,xcfar ,xd,xainfo,xbnearinfo,xcfarinfo ,xdinfo,bias,NULL,ia, ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xbfar ,xcnear,xd,xainfo,xbfarinfo ,xcnearinfo,xdinfo,bias,NULL,ia, iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xa,xbfar ,xcfar ,xd,xainfo,xbfarinfo ,xcfarinfo ,xdinfo,bias,NULL,ia, iib,iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xbfarpresent && xdfarpresent )
    {
        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const SparseVector<gentype> &xdnear = xd.nearref();
        const SparseVector<gentype> &xdfar  = xd.farref();

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        const vecInfo &xdnearinfo = xdinfo(0,-1);
        const vecInfo &xdfarinfo  = xdinfo(1,-1);

        int iib = -(((ib+1)*100)+1);
        int iid = -(((id+1)*100)+1);

        T tmp;

               yyycK4(res,xa,xbnear,xc,xdnear,xainfo,xbnearinfo,xcinfo,xdnearinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xbnear,xc,xdfar ,xainfo,xbnearinfo,xcinfo,xdfarinfo ,bias,NULL,ia, ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xbfar ,xc,xdnear,xainfo,xbfarinfo ,xcinfo,xdnearinfo,bias,NULL,ia, iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xa,xbfar ,xc,xdfar ,xainfo,xbfarinfo ,xcinfo,xdfarinfo ,bias,NULL,ia, iib,ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xcfarpresent && xdfarpresent )
    {
        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const SparseVector<gentype> &xdnear = xd.nearref();
        const SparseVector<gentype> &xdfar  = xd.farref();

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        const vecInfo &xdnearinfo = xdinfo(0,-1);
        const vecInfo &xdfarinfo  = xdinfo(1,-1);

        int iic = -(((ic+1)*100)+1);
        int iid = -(((id+1)*100)+1);

        T tmp;

               yyycK4(res,xa,xb,xcnear,xdnear,xainfo,xbinfo,xcnearinfo,xdnearinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xb,xcnear,xdfar ,xainfo,xbinfo,xcnearinfo,xdfarinfo ,bias,NULL,ia, ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xb,xcfar ,xdnear,xainfo,xbinfo,xcfarinfo ,xdnearinfo,bias,NULL,ia, ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res += yyycK4(tmp,xa,xb,xcfar ,xdfar ,xainfo,xbinfo,xcfarinfo ,xdfarinfo ,bias,NULL,ia, ib, iic,iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xafarpresent )
    {
        const SparseVector<gentype> &xanear = xa.nearref();
        const SparseVector<gentype> &xafar  = xa.farref();

        const vecInfo &xanearinfo = xainfo(0,-1);
        const vecInfo &xafarinfo  = xainfo(1,-1);

        int iia = -(((ia+1)*100)+1);

        T tmp;

               yyycK4(res,xanear,xb,xc,xd,xanearinfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xafar ,xb,xc,xd,xafarinfo ,xbinfo,xcinfo,xdinfo,bias,NULL,iia,ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xbfarpresent )
    {
        const SparseVector<gentype> &xbnear = xb.nearref();
        const SparseVector<gentype> &xbfar  = xb.farref();

        const vecInfo &xbnearinfo = xbinfo(0,-1);
        const vecInfo &xbfarinfo  = xbinfo(1,-1);

        int iib = -(((ib+1)*100)+1);

        T tmp;

               yyycK4(res,xa,xbnear,xc,xd,xainfo,xbnearinfo,xcinfo,xdinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xbfar ,xc,xd,xainfo,xbfarinfo ,xcinfo,xdinfo,bias,NULL,ia, iib,ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xcfarpresent )
    {
        const SparseVector<gentype> &xcnear = xc.nearref();
        const SparseVector<gentype> &xcfar  = xc.farref();

        const vecInfo &xcnearinfo = xcinfo(0,-1);
        const vecInfo &xcfarinfo  = xcinfo(1,-1);

        int iic = -(((ic+1)*100)+1);

        T tmp;

               yyycK4(res,xa,xb,xcnear,xd,xainfo,xbinfo,xcnearinfo,xdinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xb,xcfar ,xd,xainfo,xbinfo,xcfarinfo ,xdinfo,bias,NULL,ia, ib, iic,id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else if ( xdfarpresent )
    {
        const SparseVector<gentype> &xdnear = xd.nearref();
        const SparseVector<gentype> &xdfar  = xd.farref();

        const vecInfo &xdnearinfo = xdinfo(0,-1);
        const vecInfo &xdfarinfo  = xdinfo(1,-1);

        int iid = -(((id+1)*100)+1);

        T tmp;

               yyycK4(res,xa,xb,xc,xdnear,xainfo,xbinfo,xcinfo,xdnearinfo,bias,NULL,ia, ib, ic, id ,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
        res -= yyycK4(tmp,xa,xb,xc,xdfar ,xainfo,xbinfo,xcinfo,xdfarinfo ,bias,NULL,ia, ib, ic, iid,xdim,xconsist,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    else
    {
        yyycK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,iaset,ibset,icset,idset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyybKm(int m, T &res,
                    Vector<const SparseVector<gentype> *> &x,
                    Vector<const vecInfo *> &xinfo,
                    const T &bias,
                    Vector<int> &i,
                    const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, 
                    const Matrix<double> *xy, const Vector<int> *iiset, int assumreal, int justcalcip) const
{
    int ii;

    if ( x.size() )
    {
        int hasrank = 0;

        for ( ii = 0 ; ii < x.size() ; ii++ )
        {
            int xfarpresent = (*(x(ii))).isfaroffindpresent() ? 1 : 0;

            if ( xfarpresent )
            {
                NiceAssert( !justcalcip );

                hasrank = 1;
                break;
            }
        }

        if ( hasrank )
        {
            NiceAssert( !justcalcip );

            Vector<int> imask(i);

            imask = zeroint();

            return yyybKmb(m,res,x,xinfo,bias,i,imask,pxyprod,xdim,xconsist,resmode,mlid,xy,iiset,assumreal,justcalcip);
        }

        else
        {
            yyycKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,iiset,assumreal,justcalcip);
        }
    }

    else
    {
        yyycKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,iiset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyybKmb(int m, T &res,
                    Vector<const SparseVector<gentype> *> &x,
                    Vector<const vecInfo *> &xinfo,
                    const T &bias,
                    Vector<int> &i, Vector<int> &imask,
                    const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, 
                    const Matrix<double> *xy, const Vector<int> *iiset, int assumreal, int justcalcip) const
{
    NiceAssert( !justcalcip );

    int ii;

    // Can assume x.size() and hasrank

    if ( x.size() )
    {
        for ( ii = x.size()-1 ; ii >= 0 ; ii-- )
        {
            int xfarpresent = (*(x(ii))).isfaroffindpresent() ? 1 : 0;

            if ( xfarpresent && !imask(ii) )
            {
                Vector<const SparseVector<gentype> *> xq(x);
                Vector<const vecInfo *> xqinfo(xinfo);
                Vector<int> iq(i);
                Vector<int> iqmask(imask);

                T tmp;

                // make sure we mask out this index to stop infinite recursion

                xq("&",ii)     = &((*(x(ii))).nearref());
                xqinfo("&",ii) = &((*(xinfo(ii)))(0,-1));
                iq("&",ii)     = i(ii);
                iqmask("&",ii) = 1;

                       yyybKmb(m,res,xq,xqinfo,bias,iq,iqmask,NULL,xdim,xconsist,resmode,mlid,NULL,iiset,assumreal,justcalcip);

                xq("&",ii)     = &((*(x(ii))).farref());
                xqinfo("&",ii) = &((*(xinfo(ii)))(1,-1));
                iq("&",ii)     = -(((i(ii)+1)*100)+1);
                iqmask("&",ii) = 1;

                res -= yyybKmb(m,tmp,xq,xqinfo,bias,iq,iqmask,NULL,xdim,xconsist,resmode,mlid,NULL,iiset,assumreal,justcalcip);

                return res;
            }
        }
    }

    return yyycKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,iiset,assumreal,justcalcip);
}
















// Pre-process Gradients



// Kronecker product:
//
// res = b(0) \otimes b(1) ...
// where b(i) has dim dim.
// use NULLs to indicate vectorised identity matrix.  These are labelled (paired) as negative indices in the nn vector.
//
// Second form: dissimilar dimensions dima and dimb

template <class T>
void kronprod(SparseVector<T> &res, Vector<const SparseVector<T> *> &b, const Vector<int> &nn, int dim);

template <class T>
void kronprod(SparseVector<T> &res, Vector<const SparseVector<T> *> &b, const Vector<int> &nn, int dim)
{
    int i,j,k,l,p;
    int n = b.size();
    int ressize = (int) pow(dim,n);

    SparseVector<int> idset;

    res.zero();

    for ( i = 0 ; i < ressize ; i++ )
    {
        res("&",i) = 1.0;

        k = i;
        l = ressize;

        idset.zero();

        for ( j = 0 ; j < n ; j++ )
        {
            l = l/dim;
            p = k/l;
            k = k%l;

            if ( b(j) )
            {
                res("&",i) *= (*b(j))(p);
            }

            else
            {
                NiceAssert( nn(j) < 0 );

                if ( !(idset.isindpresent(-nn(j))) )
                {
                    idset("&",-nn(j)) = p; // store first index of kronecker-delta, but don't do anything yet
                }

                else if ( idset(-nn(j)) != p )
                {
                    res("&",i) = 0.0; // second index doesn't match first, so result is zero
                    break; // break out of inner for loop for speed
                }
            }
        }
    }

    return;
}

// Functions to allow different types to be treated "as if" they were matrices or vectors

inline double  &resizeZeroMat(double  &x, int i, int j);
inline gentype &resizeZeroMat(gentype &x, int i, int j);

inline double  &resizeZeroVec(double  &x, int i);
inline gentype &resizeZeroVec(gentype &x, int i);

inline double  &getMatElm(double  &x, int i, int j);
inline gentype &getMatElm(gentype &x, int i, int j);

inline double  &getVecElm(double  &x, int i);
inline gentype &getVecElm(gentype &x, int i);

inline double &resizeZeroMat(double &x, int i, int j)
{
    if ( ( i != 1 ) || ( j != 1 ) )
    {
        throw("Attempt to resize double as matrix");
    }

    return x = 0.0;
}

inline gentype &resizeZeroMat(gentype &x, int i, int j)
{
    x.force_matrix().resize(i,j);
    x.dir_matrix().zero();

    return x;
}

inline double &resizeZeroVec(double &x, int i)
{
    if ( i != 1 )
    {
        throw("Attempt to resize double as vector");
    }

    return x = 0.0;
}

inline gentype &resizeZeroVec(gentype &x, int i)
{
    x.force_vector().resize(i);
    x.dir_vector().zero();

    return x;
}

inline double &getMatElm(double &x, int i, int j)
{
    if ( ( i != 0 ) || ( j != 0 ) )
    {
        throw("Attempt to dereference double as matrix");
    }

    return x;
}

inline gentype &getMatElm(gentype &x, int i, int j)
{
    return x("&",i,j);
}

inline double &getVecElm(double &x, int i)
{
    if ( i != 0 )
    {
        throw("Attempt to dereference double as vector");
    }

    return x;
}

inline gentype &getVecElm(gentype &x, int i)
{
    return x("&",i);
}



template <class T> 
void MercerKernel::yyycdKK2( T &xygrad, T &xnormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset) const
{
#ifndef NDEBUG
    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xbfarfarpresent = xb.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xbind6present = xb.isfarfarfarindpresent(6) && !(xb(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );
    int xbgradOrder = xbind6present ? ( (int) xb.fff(6) ) : ( xbfarfarpresent ? 1 : 0 );

    NiceAssert( !xagradOrder );
    NiceAssert( !xbgradOrder );
#endif

    xdKK2(xygrad,xnormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    return;
}

template <class T> 
void MercerKernel::yyycd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset) const
{
#ifndef NDEBUG
    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xbfarfarpresent = xb.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xbind6present = xb.isfarfarfarindpresent(6) && !(xb(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );
    int xbgradOrder = xbind6present ? ( (int) xb.fff(6) ) : ( xbfarfarpresent ? 1 : 0 );

    NiceAssert( !xagradOrder );
    NiceAssert( !xbgradOrder );
#endif

    xd2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    return;
}

template <class T> 
void MercerKernel::yyycdnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, const T &bias, const gentype **pxyprod, int ia, int ib, int xdim, int xconsist, int assumreal, int mlid, const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset) const
{
#ifndef NDEBUG
    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xbfarfarpresent = xb.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xbind6present = xb.isfarfarfarindpresent(6) && !(xb(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );
    int xbgradOrder = xbind6present ? ( (int) xb.fff(6) ) : ( xbfarfarpresent ? 1 : 0 );

    NiceAssert( !xagradOrder );
    NiceAssert( !xbgradOrder );
#endif

    xdnKK2del(sc,n,minmaxind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    return;
}





template <class T>
void MercerKernel::qqqdK2delx(T &xscaleres, T &yscaleres,  int &minmaxind,
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          const T &bias, 
                          const gentype **pxyprod,
                          int i, int j, 
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int iaset, int ibset, int assumreal) const
{
    T xygrad;
    T xnormgrad;

    xdKK2(xygrad,xnormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,1,iaset,ibset);

    xscaleres  = xnormgrad;
    xscaleres *= 2.0;
    yscaleres  = xygrad;

    return;
}

template <class T>
void MercerKernel::qqqdnK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, 
                          const Vector<int> &q, 
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          const T &bias, 
                          const gentype **pxyprod, 
                          int i, int j, 
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int deepDerive, int iaset, int ibset, int assumreal) const
{
    int z = 0;

    if ( q.size() == 0 )
    {
        // "no gradient" case

        sc.resize(1);
        n.resize(1);

        n("&",z).resize(z);

        yyyKK2(sc("&",z),x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,0,mlid,xy00,xy10,xy11,iaset,ibset,assumreal,0);
    }

    else if ( q.size() == 1 )
    {
        if ( q(z) == 0 )
        {
            // d/dx case - result is sc(0).x + sc(1).y

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);
            
            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            qqqdK2delx(sc("&",z),sc("&",1),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,iaset,ibset,assumreal);
        }

        else
        {
            // d/dy case - result is sc(0).x + sc(1).y
            // We assume symmetry to evaluate this

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);
            
            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            qqqdK2delx(sc("&",1),sc("&",z),minmaxind,y,x,yinfo,xinfo,bias,NULL,j,i,xdim,xconsist,mlid,NULL,NULL,NULL,iaset,ibset,assumreal);
        }
    }

    else if ( q.size() == 2 )
    {
        if ( ( q(z) == 0 ) && ( q(1) == 0 ) )
        {
            // d^2/dx^2 case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            qqqd2K2delxdelx(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,iaset,ibset,assumreal);
        }

        else if ( ( q(z) == 0 ) && ( q(1) == 1 ) )
        {
            // d/dx d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            qqqd2K2delxdely(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,iaset,ibset,assumreal);
        }

        else if ( ( q(z) == 1 ) && ( q(1) == 0 ) )
        {
            // d/dy d/dx case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            qqqd2K2delxdely(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,NULL,j,i,xdim,xconsist,mlid,NULL,NULL,NULL,deepDerive,iaset,ibset,assumreal);
        }

        else
        {
            // d/dy d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);
            
            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            qqqd2K2delxdelx(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,NULL,j,i,xdim,xconsist,mlid,NULL,NULL,NULL,deepDerive,iaset,ibset,assumreal);
        }
    }

    else
    {
        xdnKK2del(sc,n,minmaxind,q,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDerive,iaset,ibset);
    }

    return;
}

template <class T>
void MercerKernel::qqqd2K2delxdelx(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 const T &bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dx_idx_j = d2K/dada da/dx_i 2x_j + d2K/dzda dz/dx_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz da/dx_i y_j + d2K/dzdz dz/dx_i y_j
    //              = d2K/dada 2x_i 2x_j + d2K/dzda y_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz 2x_i y_j + d2K/dzdz y_i y_j
    //
    // d2K/dxdx = 4 d2K/dada x.x' + 2 d2K/dzda y.x' + 2 d2K/dadz x.y' + d2K/dzdz y.y' + 2 dK/da I

    T xygrad;
    T xnormgrad;
    T xyxygrad;
    T xyxnormgrad;
    T xyynormgrad;
    T xnormxnormgrad;
    T xnormynormgrad;
    T ynormynormgrad;

    xd2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    xxscaleres = 4.0*xnormxnormgrad;
    xyscaleres = 2.0*xyxnormgrad;
    yxscaleres = xyscaleres;
    yyscaleres = xyxygrad;
    constres   = 2.0*xnormgrad;

    return;
}

template <class T>
void MercerKernel::qqqd2K2delxdely(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 const T &bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dy_idx_j = d2K/dzda dz/dy_i 2x_j + d2K/dbda db/dy_i 2x_j + d2K/dzdz dz/dy_i y_j + d2K/dbdz db/dy_i y_j + dK/dz delta_{ij}
    //              = d2K/dzda x_i     2x_j + d2K/dbda 2y_i    2x_j + d2K/dzdz x_i     y_j + d2K/dbdz 2y_i    y_j + dK/dz delta_{ij}
    //              = 2 d2K/dzda x_i x_j + 4 d2K/dbda y_i x_j + d2K/dzdz x_i y_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dx_idy_j = 2 d2K/dzda x_i x_j + 4 d2K/dbda x_i y_j + d2K/dzdz y_i x_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dxdy = 2 d2K/dzda x.x' + 4 d2K/dbda x.y' + d2K/dzdz y.x' + 2 d2K/dbdz y.y' + dK/dz I

    T xygrad;
    T xnormgrad;
    T xyxygrad;
    T xyxnormgrad;
    T xyynormgrad;
    T xnormxnormgrad;
    T xnormynormgrad;
    T ynormynormgrad;

    xd2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    xxscaleres = 2.0*xyxnormgrad;
    xyscaleres = 4.0*xnormynormgrad;
    yxscaleres = xyxygrad;
    yyscaleres = 2.0*xyynormgrad;
    constres   = xygrad;

    return;
}







template <class T>
T &MercerKernel::yyycK0(T &res,
                    const T &bias,
                    const gentype **pxyprod,
                    int xdim, int xconsist, int xresmode, int mlid, int assumreal, int justcalcip) const
{
    return yyyKK0(res,bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,justcalcip);
}

template <class T>
T &MercerKernel::yyycK1(T &res,
                    const SparseVector<gentype> &xa, 
                    const vecInfo &xainfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, 
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy00, int iaset, int assumreal, int justcalcip) const
{
#ifndef NDEBUG
    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );

    NiceAssert( !xagradOrder )
#endif

    yyyKK1(res,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,resmode,mlid,xy00,iaset,assumreal,justcalcip);

    return res;
}

template <class T>
T &MercerKernel::yyycK2(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                    const vecInfo &xainfo, const vecInfo &xbinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib,
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, int iaset, int ibset, int assumreal, int justcalcip) const
{
    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xbfarfarpresent = xb.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xbind6present = xb.isfarfarfarindpresent(6) && !(xb(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );
    int xbgradOrder = xbind6present ? ( (int) xb.fff(6) ) : ( xbfarfarpresent ? 1 : 0 );

    int dummyind = -1;

    if ( xagradOrder && xbgradOrder )
    {
        NiceAssert( !justcalcip );

        const SparseVector<gentype> &xanear   = xa.nearref();
        const SparseVector<gentype> &xafarfar = xa.farfarref();

        const SparseVector<gentype> &xbnear   = xb.nearref();
        const SparseVector<gentype> &xbfarfar = xb.farfarref();

        if ( xafarfarpresent && xbfarfarpresent )
        {
            // case 22

            if ( ( xagradOrder == 1 ) && ( xbgradOrder == 1 ) )
            {
                // <ex,d2K/dxdy,ey>

                T xascaleres;
                T xyscaleres;
                T yxscaleres;
                T xbscaleres;

                T constres;

                qqqd2K2delxdely(xascaleres,xbscaleres,xyscaleres,yxscaleres,constres,dummyind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                gentype exa;
                gentype exy;
                gentype xey;
                gentype yey;

                gentype exey;

                twoProduct(exa,xafarfar,xanear);
                twoProduct(exy,xafarfar,xbnear);
                twoProduct(xey,xanear,xbfarfar);
                twoProduct(yey,xbnear,xbfarfar);

                twoProduct(exey,xafarfar,xbfarfar);

                res = (((T) exa)*xascaleres*((T) xey)) 
                    + (((T) exa)*xyscaleres*((T) yey)) 
                    + (((T) exy)*yxscaleres*((T) xey)) 
                    + (((T) exy)*xbscaleres*((T) yey)) 
                    + (constres*((T) exey));
            }

            else
            {
                // <(ex d/dx)^n (ey d/dy)^n,K>

                int resxadim = (int) pow(xdim,xagradOrder);
                int resxbdim = (int) pow(xdim,xbgradOrder);

                res = 0.0;

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(xagradOrder+xagradOrder);

                retVector<int> tmpva;

                q("&",0          ,1,xagradOrder-1            ,tmpva) = 0;
                q("&",xagradOrder,1,xagradOrder+xbgradOrder-1,tmpva) = 1;

                // Here we use the result: vec(ABC) = vec(C' \otimes A).vec(B), so if
                // we let a = xifarfar, c = xjfarfar, then:
                // a'.B.c = vec(c' \otimes a').vec(B)
                //        = vec(c \otimes a)'.vec(B)
                // where vec(B) is the vectorised *transpose* gradient we usually calculate, so, getting rid of the transpose,
                // we need to take the inner product with vec(ex \otimes ey)
                // ORIGINAL INCORRECT VERSION: where vec(B) is precisely the vectorised gradient we usually calculate
                //
                // CLARIFICATION: see pdf in stable bayesian optimisation paper

                int iqa,jqa;
                gentype tmp;

                SparseVector<gentype> kres;
                SparseVector<gentype> farfarprod;
                Vector<const SparseVector<gentype> *> bord;

                // first calculate c \otimes a

                int dimmy = 0;

                //kronprod(farfarprod,*xjfarfar,*xifarfar,pow(xdim,xbgradOrder),pow(xdim,xagradOrder)); //- NB order wrong here, have tested and confirmed, following line is correct.
                kronprod(farfarprod,dimmy,xafarfar,xbfarfar,resxadim,resxbdim);

                // Gradient

                qqqdnK2del(sc,nn,dummyind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                // Then proceed in fully vectorised form

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? &xanear : ( ( nn(iqa)(jqa) == 1 ) ? &xbnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdim);

                    twoProduct(tmp,farfarprod,kres);

                    res += ((T) tmp)*((T) sc(iqa));
                }
            }
        }

        else if ( xafarfarpresent && !xbfarfarpresent )
        {
            // case 42

            if ( ( xagradOrder == 1 ) && ( xbgradOrder == 1 ) )
            {
                // <ex,d2K/dxdy,ey>

                resizeZeroVec(res,xdim);

                T xascaleres;
                T xyscaleres;
                T yxscaleres;
                T xbscaleres;

                T constres;

                int jqa;

                qqqd2K2delxdely(xascaleres,xbscaleres,xyscaleres,yxscaleres,constres,dummyind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                gentype exa;
                gentype exy;

                twoProduct(exa,xafarfar,xanear);
                twoProduct(exy,xafarfar,xbnear);

                for ( jqa = 0 ; jqa < xdim ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) exa)*((T) xascaleres)*((T) (xanear(jqa)));
                    getVecElm(res,jqa) += ((T) exa)*((T) xyscaleres)*((T) (xbnear(jqa)));
                    getVecElm(res,jqa) += ((T) exy)*((T) yxscaleres)*((T) (xanear(jqa)));
                    getVecElm(res,jqa) += ((T) exy)*((T) xbscaleres)*((T) (xbnear(jqa)));

                    getVecElm(res,jqa) += ((T) constres)*((T) (xafarfar(jqa)));
                }
            }

            else
            {
                // <(ex d/dx)^n (ey d/dy)^n,K>

                int resxadim = (int) pow(xdim,xagradOrder);
                int resxbdim = (int) pow(xdim,xbgradOrder);

                resizeZeroVec(res,resxbdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(xagradOrder+xbgradOrder);

                retVector<int> tmpva;

                q("&",0          ,1,xagradOrder-1            ,tmpva) = 0;
                q("&",xagradOrder,1,xagradOrder+xbgradOrder-1,tmpva) = 1;

                // Here we use the result: vec(ABC) = vec(C' \otimes A).vec(B), so if
                // we let a = xifarfar, c = xjfarfar, then:
                // a'.B.c = vec(c' \otimes a').vec(B)
                //        = vec(c \otimes a)'.vec(B)
                // where vec(B) is the vectorised *transpose* gradient we usually calculate, so, getting rid of the transpose,
                // we need to take the inner product with vec(ex \otimes ey)
                // ORIGINAL INCORRECT VERSION: where vec(B) is precisely the vectorised gradient we usually calculate
                //
                // CLARIFICATION: see pdf in stable bayesian optimisation paper

                int iqa,jqa,kqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                // Gradient

                qqqdnK2del(sc,nn,dummyind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                // Then proceed in fully vectorised form

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? &xanear : ( ( nn(iqa)(jqa) == 1 ) ? &xbnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdim);

                    for ( jqa = 0 ; jqa < resxadim ; jqa++ )
                    {
                        for ( kqa = 0 ; kqa < resxbdim ; kqa++ )
                        {
                            getVecElm(res,kqa) += ((T) sc(iqa))*((T) (kres((jqa*resxbdim)+kqa)))*((T) (xafarfar)(jqa));
                        }
                    }
                }
            }
        }

        else if ( !xafarfarpresent && xbfarfarpresent )
        {
            // case 24

            if ( ( xagradOrder == 1 ) && ( xbgradOrder == 1 ) )
            {
                // <ex,d2K/dxdy,ey>

                resizeZeroVec(res,xdim);

                T xascaleres;
                T xyscaleres;
                T yxscaleres;
                T xbscaleres;

                T constres;

                int jqa;

                qqqd2K2delxdely(xascaleres,xbscaleres,xyscaleres,yxscaleres,constres,dummyind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                gentype xey;
                gentype yey;

                twoProduct(xey,xanear,xbfarfar);
                twoProduct(yey,xbnear,xbfarfar);

                for ( jqa = 0 ; jqa < xdim ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) ((xanear)(jqa)))*((T) xascaleres)*((T) xey);
                    getVecElm(res,jqa) += ((T) ((xanear)(jqa)))*((T) xyscaleres)*((T) yey);
                    getVecElm(res,jqa) += ((T) ((xbnear)(jqa)))*((T) yxscaleres)*((T) xey);
                    getVecElm(res,jqa) += ((T) ((xbnear)(jqa)))*((T) xbscaleres)*((T) yey);

                    getVecElm(res,jqa) += ((T) constres)*((T) ((xbfarfar)(jqa)));
                }
            }

            else
            {
                // <(ex d/dx)^n (ey d/dy)^n,K>

                int resxadim = (int) pow(xdim,xagradOrder);
                int resxbdim = (int) pow(xdim,xbgradOrder);

                resizeZeroVec(res,resxadim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(xagradOrder+xbgradOrder);

                retVector<int> tmpva;

                q("&",0          ,1,xagradOrder-1            ,tmpva) = 0;
                q("&",xagradOrder,1,xagradOrder+xbgradOrder-1,tmpva) = 1;

                // Here we use the result: vec(ABC) = vec(C' \otimes A).vec(B), so if
                // we let a = xifarfar, c = xjfarfar, then:
                // a'.B.c = vec(c' \otimes a').vec(B)
                //        = vec(c \otimes a)'.vec(B)
                // where vec(B) is the vectorised *transpose* gradient we usually calculate, so, getting rid of the transpose,
                // we need to take the inner product with vec(ex \otimes ey)
                // ORIGINAL INCORRECT VERSION: where vec(B) is precisely the vectorised gradient we usually calculate
                //
                // CLARIFICATION: see pdf in stable bayesian optimisation paper

                int iqa,jqa,kqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                // Gradient

                qqqdnK2del(sc,nn,dummyind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                // Then proceed in fully vectorised form

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? &xanear : ( ( nn(iqa)(jqa) == 1 ) ? &xbnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdim);

                    for ( jqa = 0 ; jqa < resxadim ; jqa++ )
                    {
                        for ( kqa = 0 ; kqa < resxbdim ; kqa++ )
                        {
                            getVecElm(res,jqa) += ((T) sc(iqa))*((T) (kres((jqa*resxbdim)+kqa)))*((T) (xbfarfar)(kqa));
                        }
                    }
                }
            }
        }

        else
        {
            // case 44

            if ( ( xagradOrder == 1 ) && ( xbgradOrder == 1 ) )
            {
                // <ex,d2K/dxdy,ey>

                resizeZeroMat(res,xdim,xdim);

                T xascaleres;
                T xyscaleres;
                T yxscaleres;
                T xbscaleres;

                T constres;

                int jqa,kqa;

                qqqd2K2delxdely(xascaleres,xbscaleres,xyscaleres,yxscaleres,constres,dummyind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdim ; jqa++ )
                {
                    for ( kqa = 0 ; kqa < xdim ; kqa++ )
                    {
                        getMatElm(res,jqa,kqa) += ((T) xascaleres)*((T) ((xanear)(jqa)))*((T) ((xanear)(kqa)));
                        getMatElm(res,jqa,kqa) += ((T) xbscaleres)*((T) ((xbnear)(jqa)))*((T) ((xbnear)(kqa)));
                        getMatElm(res,jqa,kqa) += ((T) xyscaleres)*((T) ((xanear)(jqa)))*((T) ((xbnear)(kqa)));
                        getMatElm(res,jqa,kqa) += ((T) yxscaleres)*((T) ((xbnear)(jqa)))*((T) ((xanear)(kqa)));
                    }

                    getMatElm(res,jqa,jqa) += ((T) constres);
                }
            }

            else
            {
                // <(ex d/dx)^n (ey d/dy)^n,K>

                int resxadim = (int) pow(xdim,xagradOrder);
                int resxbdim = (int) pow(xdim,xbgradOrder);

                resizeZeroMat(res,resxadim,resxbdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(xagradOrder+xbgradOrder);

                retVector<int> tmpva;

                q("&",0          ,1,xagradOrder-1            ,tmpva) = 0;
                q("&",xagradOrder,1,xagradOrder+xbgradOrder-1,tmpva) = 1;

                // Here we use the result: vec(ABC) = vec(C' \otimes A).vec(B), so if
                // we let a = xifarfar, c = xjfarfar, then:
                // a'.B.c = vec(c' \otimes a').vec(B)
                //        = vec(c \otimes a)'.vec(B)
                // where vec(B) is the vectorised *transpose* gradient we usually calculate, so, getting rid of the transpose,
                // we need to take the inner product with vec(ex \otimes ey)
                // ORIGINAL INCORRECT VERSION: where vec(B) is precisely the vectorised gradient we usually calculate
                //
                // CLARIFICATION: see pdf in stable bayesian optimisation paper

                int iqa,jqa,kqa;

                SparseVector<gentype> kres;
                SparseVector<gentype> farfarprod;
                Vector<const SparseVector<gentype> *> bord;

                // Gradient

                qqqdnK2del(sc,nn,dummyind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                // Then proceed in fully vectorised form

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? &xanear : ( ( nn(iqa)(jqa) == 1 ) ? &xbnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdim);

                    for ( jqa = 0 ; jqa < resxadim ; jqa++ )
                    {
                        for ( kqa = 0 ; kqa < resxbdim ; kqa++ )
                        {
                            getMatElm(res,jqa,kqa) += ((T) sc(iqa))*((T) (kres((jqa*resxbdim)+kqa)));
                        }
                    }
                }
            }
        }
    }

    else if ( xagradOrder && !xbgradOrder )
    {
        NiceAssert( !justcalcip );

        const SparseVector<gentype> &xanear   = xa.nearref();
        const SparseVector<gentype> &xafarfar = xa.farfarref();

        const SparseVector<gentype> &xbnear = xb.nearref();

        if ( xafarfarpresent )
        {
            // case 2

            if ( xagradOrder == 1 )
            {
                // <ex,dK/dx> = <ex,x.ax + y.ay>
                //            = <ex,x>.ax + <ex,y>.ay

                T xascaleres; 
                T xbscaleres;

                qqqdK2delx(xascaleres,xbscaleres,dummyind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                gentype exa;
                gentype exy;

                twoProduct(exa,xafarfar,xanear);
                twoProduct(exy,xafarfar,xbnear);

                res = (((T) exa)*xascaleres)+(((T) exy)*xbscaleres);
            }

            else
            {
                // <(ex d/dx)^n,K>

                res = 0.0;

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(xagradOrder);

                q = zeroint();

                int iqa,jqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                qqqdnK2del(sc,nn,dummyind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? &xanear : ( ( nn(iqa)(jqa) == 1 ) ? &xbnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdim);

                    twoProduct(tmp,xafarfar,kres);

                    res += ((T) tmp)*((T) sc(iqa));
                }
            }
        }

        else
        {
            // case 4

            if ( xagradOrder == 1 )
            {
                // <ex,dK/dx> = <ex,x.ax + y.ay>
                //            = <ex,x>.ax + <ex,y>.ay

                resizeZeroVec(res,xdim);

                T xascaleres; 
                T xbscaleres;

                int jqa;

                qqqdK2delx(xascaleres,xbscaleres,dummyind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdim ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) xascaleres)*((T) ((xanear)(jqa)));
                    getVecElm(res,jqa) += ((T) xbscaleres)*((T) ((xbnear)(jqa)));
                }
            }

            else
            {
                // <(ex d/dx)^n,K>

                int resdim = (int) pow(xdim,xagradOrder);

                resizeZeroVec(res,resdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(xagradOrder);

                q = zeroint();

                int iqa,jqa;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                qqqdnK2del(sc,nn,dummyind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? &xanear : ( ( nn(iqa)(jqa) == 1 ) ? &xbnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdim);

                    for ( jqa = 0 ; jqa < resdim ; jqa++ )
                    {
                        getVecElm(res,jqa) += ((T) sc(iqa))*((T) kres(jqa));
                    }
                }
            }
        }
    }

    else if ( !xagradOrder && xbgradOrder )
    {
        NiceAssert( !justcalcip );

        const SparseVector<gentype> &xanear = xa.nearref();

        const SparseVector<gentype> &xbnear   = xb.nearref();
        const SparseVector<gentype> &xbfarfar = xb.farfarref();

        if ( xbfarfarpresent )
        {
            // case 20

            if ( xbgradOrder == 1 )
            {
                // <dK/dy,ey> = <xa.x + ya.y,ey>
                //            = xa.<x,ey> + ya.<y,ey>

                T xascaleres;
                T xbscaleres;

                // MOD: do by reversing x and y, assuming symmetry.
                //dK2dely(xascaleres,xbscaleres,dummyind,ia,ib,bias,altK,NULL,xanear,xbnear,xanearinfo,xbnearinfo,iaset,ibset);
                qqqdK2delx(xbscaleres,xascaleres,dummyind,xb,xa,xbinfo,xainfo,bias,pxyprod,ib,ia,xdim,xconsist,mlid,xy11,xy10,xy00,ibset,iaset,assumreal);

                NiceAssert( dummyind < 0 );

                gentype xey;
                gentype yey;

                twoProduct(xey,xanear,xbfarfar);
                twoProduct(yey,xbnear,xbfarfar);

                res = (xascaleres*((T) xey))+(xbscaleres*((T) yey));
            }

            else
            {
                // <(ex d/dy)^n,K>

                res = 0.0;

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(xbgradOrder);

                q = 1;

                qqqdnK2del(sc,nn,dummyind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                int iqa,jqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? &xanear : ( ( nn(iqa)(jqa) == 1 ) ? &xbnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdim);

                    twoProduct(tmp,xbfarfar,kres);

                    res += ((T) tmp)*((T) sc(iqa));
                }
            }
        }

        else
        {
            // case 40

            if ( xbgradOrder == 1 )
            {
                // <dK/dy,ey> = <xa.x + ya.y,ey>
                //            = xa.<x,ey> + ya.<y,ey>

                resizeZeroVec(res,xdim);

                T xascaleres;
                T xbscaleres;

                int jqa;

                // MOD: do by reversing x and y, assuming symmetry.
                //dK2dely(xscaleres,yscaleres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,iaset,ibset);
                qqqdK2delx(xbscaleres,xascaleres,dummyind,xb,xa,xbinfo,xainfo,bias,pxyprod,ib,ia,xdim,xconsist,mlid,xy11,xy10,xy00,ibset,iaset,assumreal);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdim ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) xascaleres)*((T) ((xanear)(jqa)));
                    getVecElm(res,jqa) += ((T) xbscaleres)*((T) ((xbnear)(jqa)));
                }
            }

            else
            {
                // <(ex d/dy)^n,K>

                int resdim = (int) pow(xdim,xbgradOrder);

                resizeZeroVec(res,resdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(xbgradOrder);

                q = 1;

                qqqdnK2del(sc,nn,dummyind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,mlid,xy00,xy10,xy11,0,iaset,ibset,assumreal);

                NiceAssert( dummyind < 0 );

                int iqa,jqa;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? &xanear : ( ( nn(iqa)(jqa) == 1 ) ? &xbnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdim);

                    for ( jqa = 0 ; jqa < resdim ; jqa++ )
                    {
                        getVecElm(res,jqa) += ((T) sc(iqa))*((T) kres(jqa));
                    }
                }
            }
        }
    }

    else
    {
        yyyKK2(res,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,iaset,ibset,assumreal,justcalcip);
    }

    return res;
}

template <class T>
T &MercerKernel::yyycK3(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iaset, int ibset, int icset, int assumreal, int justcalcip) const
{
#ifndef NDEBUG
    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xbfarfarpresent = xb.isfarfaroffindpresent() ? 1 : 0;
    int xcfarfarpresent = xc.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xbind6present = xb.isfarfarfarindpresent(6) && !(xb(6).isValNull());
    int xcind6present = xc.isfarfarfarindpresent(6) && !(xc(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );
    int xbgradOrder = xbind6present ? ( (int) xb.fff(6) ) : ( xbfarfarpresent ? 1 : 0 );
    int xcgradOrder = xcind6present ? ( (int) xc.fff(6) ) : ( xcfarfarpresent ? 1 : 0 );

    NiceAssert( !xagradOrder && !xbgradOrder && !xcgradOrder );
#endif

    yyyKK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,iaset,ibset,icset,assumreal,justcalcip);

    return res;
}

template <class T>
T &MercerKernel::yyycK4(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, int id, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int assumreal, int justcalcip) const
{
#ifndef NDEBUG
    int xafarfarpresent = xa.isfarfaroffindpresent() ? 1 : 0;
    int xbfarfarpresent = xb.isfarfaroffindpresent() ? 1 : 0;
    int xcfarfarpresent = xc.isfarfaroffindpresent() ? 1 : 0;
    int xdfarfarpresent = xd.isfarfaroffindpresent() ? 1 : 0;

    int xaind6present = xa.isfarfarfarindpresent(6) && !(xa(6).isValNull());
    int xbind6present = xb.isfarfarfarindpresent(6) && !(xb(6).isValNull());
    int xcind6present = xc.isfarfarfarindpresent(6) && !(xc(6).isValNull());
    int xdind6present = xd.isfarfarfarindpresent(6) && !(xd(6).isValNull());

    int xagradOrder = xaind6present ? ( (int) xa.fff(6) ) : ( xafarfarpresent ? 1 : 0 );
    int xbgradOrder = xbind6present ? ( (int) xb.fff(6) ) : ( xbfarfarpresent ? 1 : 0 );
    int xcgradOrder = xcind6present ? ( (int) xc.fff(6) ) : ( xcfarfarpresent ? 1 : 0 );
    int xdgradOrder = xdind6present ? ( (int) xd.fff(6) ) : ( xdfarfarpresent ? 1 : 0 );

    NiceAssert( !xagradOrder && !xbgradOrder && !xcgradOrder && !xdgradOrder );
#endif

    yyyKK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,iaset,ibset,icset,idset,assumreal,justcalcip);

    return res;
}

template <class T>
T &MercerKernel::yyycKm(int m, T &res,
                    Vector<const SparseVector<gentype> *> &x,
                    Vector<const vecInfo *> &xinfo,
                    const T &bias,
                    Vector<int> &i,
                    const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, 
                    const Matrix<double> *xy, const Vector<int> *iiset, int assumreal, int justcalcip) const
{
#ifndef NDEBUG
    int ii;

    if ( x.size() )
    {
        for ( ii = 0 ; ii < x.size() ; ii++ )
        {
            int xfarfarpresent = (*(x(ii))).isfarfaroffindpresent() ? 1 : 0;

            int xind6present = (*(x(ii))).isfarfarfarindpresent(6) && !((*(x(ii)))(6).isValNull());

            int xgradOrder = xind6present ? ( (int) (*(x(ii))).fff(6) ) : ( xfarfarpresent ? 1 : 0 );

            NiceAssert( !xgradOrder );
        }
    }
#endif

    yyyKKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,iiset,assumreal,justcalcip);

    return res;
}
















// Pre-process [ xa ~ xb ~ ... ] forms

template <class T>
T &MercerKernel::yyyKK0(T &res,
                    const T &bias,
                    const gentype **pxyprod,
                    int xdim, int xconsist, int xresmode, int mlid, int assumreal, int justcalcip) const
{
    return xKKK0(res,bias,pxyprod,xdim,xconsist,assumreal,xresmode,mlid,justcalcip);
}

template <class T>
T &MercerKernel::yyyKK1(T &res,
                    const SparseVector<gentype> &x, 
                    const vecInfo &xinfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int i, 
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy, int iset, int assumreal, int justcalcip) const
{
//    return xKKK1(res,x,xinfo,bias,pxyprod,i,xdim,xconsist,resmode,assumreal,mlid,xy,justcalcip,iset);
        int iupm = x.nearupsize();

        if ( iupm == 1 )
        {
            xKKK1(res,x,xinfo,bias,pxyprod,i,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,iset);
        }

        else if ( iupm == 2 )
        {
            const SparseVector<gentype> &rxa = x.nearrefup(0);
            const SparseVector<gentype> &rxb = x.nearrefup(1);

            const vecInfo &rxainfo = xinfo(-1,0);
            const vecInfo &rxbinfo = xinfo(-1,1);

            int ixa = i;
            int ixb = UPNTVI(i,1);

            int ixaset = iset;
            int ixbset = iset;

            xKKK2(res,rxa,rxb,rxainfo,rxbinfo,bias,NULL,ixa,ixb,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,ixaset,ixbset);
        }

        else if ( iupm == 3 )
        {
            const SparseVector<gentype> &rxa = x.nearrefup(0);
            const SparseVector<gentype> &rxb = x.nearrefup(1);
            const SparseVector<gentype> &rxc = x.nearrefup(2);

            const vecInfo &rxainfo = xinfo(-1,0);
            const vecInfo &rxbinfo = xinfo(-1,1);
            const vecInfo &rxcinfo = xinfo(-1,2);

            int ixa = i;
            int ixb = UPNTVI(i,1);
            int ixc = UPNTVI(i,2);

            int ixaset = iset;
            int ixbset = iset;
            int ixcset = iset;

            xKKK3(res,rxa,rxb,rxc,rxainfo,rxbinfo,rxcinfo,bias,NULL,ixa,ixb,ixc,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset);
        }

        else if ( iupm == 4 )
        {
            const SparseVector<gentype> &rxa = x.nearrefup(0);
            const SparseVector<gentype> &rxb = x.nearrefup(1);
            const SparseVector<gentype> &rxc = x.nearrefup(2);
            const SparseVector<gentype> &rxd = x.nearrefup(3);

            const vecInfo &rxainfo = xinfo(-1,0);
            const vecInfo &rxbinfo = xinfo(-1,1);
            const vecInfo &rxcinfo = xinfo(-1,2);
            const vecInfo &rxdinfo = xinfo(-1,3);

            int ixa = i;
            int ixb = UPNTVI(i,1);
            int ixc = UPNTVI(i,2);
            int ixd = UPNTVI(i,3);

            int ixaset = iset;
            int ixbset = iset;
            int ixcset = iset;
            int ixdset = iset;

            xKKK4(res,rxa,rxb,rxc,rxd,rxainfo,rxbinfo,rxcinfo,rxdinfo,bias,NULL,ixa,ixb,ixc,ixd,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset,ixdset);
        }

        else
        {
            int ii;

            Vector<int> iv(iupm);
            Vector<const SparseVector<gentype> *> xxv(iupm);
            Vector<const vecInfo *> xxvinfo(iupm);
            Vector<int> ivset(iupm);

            if ( iupm )
            {
                for ( ii = 0 ; ii < iupm ; ii++ )
                {
                    iv("&",ii) = ii ? UPNTVI(i,ii) : i;
                    xxv("&",ii) = &(x.nearrefup(ii));
                    xxvinfo("&",ii) = &(xinfo(-1,ii));
                    ivset("&",ii) = iset;
                }
            }

            xKKKm(iupm,res,xxv,xxvinfo,bias,iv,NULL,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,&ivset);
        }

    return res;
}

template <class T>
T &MercerKernel::yyyKK2(T &res,
                    const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                    const vecInfo &xinfo, const vecInfo &yinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int i, int j,
                    int xdim, int xconsist, int resmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, int iset, int jset, int assumreal, int justcalcip) const
{
//    return xKKK2(res,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,resmode,mlid,xy,0,iset,jset);
        int iupm = x.nearupsize();
        int jupm = y.nearupsize();

        if ( ( iupm == 1 ) && ( jupm == 1 ) )
        {
            xKKK2(res,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,justcalcip,iset,jset);
        }

        else if ( ( iupm == 1 ) && ( jupm == 2 ) )
        {
//NB: order of vectors is very important here (see splits, ns, below)
            const SparseVector<gentype> &rxa = x.nearrefup(0);
            const SparseVector<gentype> &rxb = y.nearrefup(0);
            const SparseVector<gentype> &rxc = y.nearrefup(1);

            const vecInfo &rxainfo = xinfo(-1,0);
            const vecInfo &rxbinfo = yinfo(-1,0);
            const vecInfo &rxcinfo = yinfo(-1,1);

            int ixa = i;
            int ixb = j;
            int ixc = UPNTVI(j,1);

            int ixaset = iset;
            int ixbset = jset;
            int ixcset = jset;

            xKKK3(res,rxa,rxb,rxc,rxainfo,rxbinfo,rxcinfo,bias,NULL,ixa,ixb,ixc,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset);
        }

        else if ( ( iupm == 2 ) && ( jupm == 1 ) )
        {
            const SparseVector<gentype> &rxa = x.nearrefup(0);
            const SparseVector<gentype> &rxb = x.nearrefup(1);
            const SparseVector<gentype> &rxc = y.nearrefup(0);

            const vecInfo &rxainfo = xinfo(-1,0);
            const vecInfo &rxbinfo = xinfo(-1,1);
            const vecInfo &rxcinfo = yinfo(-1,0);

            int ixa = i;
            int ixb = UPNTVI(i,1);
            int ixc = j;

            int ixaset = iset;
            int ixbset = iset;
            int ixcset = jset;

            xKKK3(res,rxa,rxb,rxc,rxainfo,rxbinfo,rxcinfo,bias,NULL,ixa,ixb,ixc,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset);
        }

        else if ( ( iupm == 1 ) && ( jupm == 3 ) )
        {
            const SparseVector<gentype> &rxa = x.nearrefup(0);
            const SparseVector<gentype> &rxb = y.nearrefup(0);
            const SparseVector<gentype> &rxc = y.nearrefup(1);
            const SparseVector<gentype> &rxd = y.nearrefup(2);

            const vecInfo &rxainfo = xinfo(-1,0);
            const vecInfo &rxbinfo = yinfo(-1,0);
            const vecInfo &rxcinfo = yinfo(-1,1);
            const vecInfo &rxdinfo = yinfo(-1,2);

            int ixa = i;
            int ixb = j;
            int ixc = UPNTVI(j,1);
            int ixd = UPNTVI(j,2);

            int ixaset = iset;
            int ixbset = jset;
            int ixcset = jset;
            int ixdset = jset;

            xKKK4(res,rxa,rxb,rxc,rxd,rxainfo,rxbinfo,rxcinfo,rxdinfo,bias,NULL,ixa,ixb,ixc,ixd,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset,ixdset);
        }

        else if ( ( iupm == 2 ) && ( jupm == 2 ) )
        {
            const SparseVector<gentype> &rxa = x.nearrefup(0);
            const SparseVector<gentype> &rxb = x.nearrefup(1);
            const SparseVector<gentype> &rxc = y.nearrefup(0);
            const SparseVector<gentype> &rxd = y.nearrefup(1);

            const vecInfo &rxainfo = xinfo(-1,0);
            const vecInfo &rxbinfo = xinfo(-1,1);
            const vecInfo &rxcinfo = yinfo(-1,0);
            const vecInfo &rxdinfo = yinfo(-1,1);

            int ixa = i;
            int ixb = UPNTVI(i,1);
            int ixc = j;
            int ixd = UPNTVI(j,1);

            int ixaset = iset;
            int ixbset = iset;
            int ixcset = jset;
            int ixdset = jset;

            xKKK4(res,rxa,rxb,rxc,rxd,rxainfo,rxbinfo,rxcinfo,rxdinfo,bias,NULL,ixa,ixb,ixc,ixd,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset,ixdset);
        }

        else if ( ( iupm == 3 ) && ( jupm == 1 ) )
        {
            const SparseVector<gentype> &rxa = x.nearrefup(0);
            const SparseVector<gentype> &rxb = x.nearrefup(1);
            const SparseVector<gentype> &rxc = x.nearrefup(2);
            const SparseVector<gentype> &rxd = y.nearrefup(0);

            const vecInfo &rxainfo = xinfo(-1,0);
            const vecInfo &rxbinfo = xinfo(-1,1);
            const vecInfo &rxcinfo = xinfo(-1,2);
            const vecInfo &rxdinfo = yinfo(-1,0);

            int ixa = i;
            int ixb = UPNTVI(i,1);
            int ixc = UPNTVI(i,2);
            int ixd = j;

            int ixaset = iset;
            int ixbset = iset;
            int ixcset = iset;
            int ixdset = jset;

            xKKK4(res,rxa,rxb,rxc,rxd,rxainfo,rxbinfo,rxcinfo,rxdinfo,bias,NULL,ixa,ixb,ixc,ixd,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset,ixdset);
        }

        else
        {
            int ii;

            Vector<int> iv(iupm+jupm);
            Vector<const SparseVector<gentype> *> xxv(iupm+jupm);
            Vector<const vecInfo *> xxvinfo(iupm+jupm);
            Vector<int> ivset(iupm+jupm);

            if ( iupm )
            {
                for ( ii = 0 ; ii < iupm ; ii++ )
                {
                    iv("&",ii) = ii ? UPNTVI(i,ii) : i;
                    xxv("&",ii) = &(x.nearrefup(ii));
                    xxvinfo("&",ii) = &(xinfo(-1,ii));
                    ivset("&",ii) = iset;
                }
            }

            if ( jupm )
            {
                for ( ii = 0 ; ii < jupm ; ii++ )
                {
                    iv("&",ii+iupm) = ii ? UPNTVI(j,ii) : j;
                    xxv("&",ii+iupm) = &(y.nearrefup(ii));
                    xxvinfo("&",ii+iupm) = &(yinfo(-1,ii));
                    ivset("&",ii+iupm) = jset;
                }
            }

            xKKKm(iupm+jupm,res,xxv,xxvinfo,bias,iv,NULL,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,&ivset);
        }


    return res;
}

template <class T>
T &MercerKernel::yyyKK3(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int iaset, int ibset, int icset, int assumreal, int justcalcip) const
{
//    return xKKK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,assumreal,xresmode,mlid,xy,0,iaset,ibset,icset);
        int aupm = xa.nearupsize();
        int bupm = xb.nearupsize();
        int cupm = xc.nearupsize();

        if ( ( aupm == 1 ) && ( bupm == 1 ) && ( cupm == 1 ) )
        {
            xKKK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,assumreal,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,justcalcip,iaset,ibset,icset);
        }

        else if ( ( aupm == 1 ) && ( bupm == 1 ) && ( cupm == 2 ) )
        {
            const SparseVector<gentype> &rxa = xa.nearrefup(0);
            const SparseVector<gentype> &rxb = xb.nearrefup(0);
            const SparseVector<gentype> &rxc = xc.nearrefup(0);
            const SparseVector<gentype> &rxd = xc.nearrefup(1);

            const vecInfo &rxainfo = xainfo(-1,0);
            const vecInfo &rxbinfo = xbinfo(-1,0);
            const vecInfo &rxcinfo = xcinfo(-1,0);
            const vecInfo &rxdinfo = xcinfo(-1,1);

            int ixa = ia;
            int ixb = ib;
            int ixc = ic;
            int ixd = UPNTVI(ic,1);

            int ixaset = iaset;
            int ixbset = ibset;
            int ixcset = icset;
            int ixdset = icset;

            xKKK4(res,rxa,rxb,rxc,rxd,rxainfo,rxbinfo,rxcinfo,rxdinfo,bias,NULL,ixa,ixb,ixc,ixd,xdim,xconsist,assumreal,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset,ixdset);
        }

        else if ( ( aupm == 1 ) && ( bupm == 2 ) && ( cupm == 1 ) )
        {
            const SparseVector<gentype> &rxa = xa.nearrefup(0);
            const SparseVector<gentype> &rxb = xb.nearrefup(0);
            const SparseVector<gentype> &rxc = xb.nearrefup(1);
            const SparseVector<gentype> &rxd = xc.nearrefup(0);

            const vecInfo &rxainfo = xainfo(-1,0);
            const vecInfo &rxbinfo = xbinfo(-1,0);
            const vecInfo &rxcinfo = xbinfo(-1,1);
            const vecInfo &rxdinfo = xcinfo(-1,0);

            int ixa = ia;
            int ixb = ib;
            int ixc = UPNTVI(ib,1);
            int ixd = ic;

            int ixaset = iaset;
            int ixbset = ibset;
            int ixcset = ibset;
            int ixdset = icset;

            xKKK4(res,rxa,rxb,rxc,rxd,rxainfo,rxbinfo,rxcinfo,rxdinfo,bias,NULL,ixa,ixb,ixc,ixd,xdim,xconsist,assumreal,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset,ixdset);
        }

        else if ( ( aupm == 2 ) && ( bupm == 1 ) && ( cupm == 1 ) )
        {
            const SparseVector<gentype> &rxa = xa.nearrefup(0);
            const SparseVector<gentype> &rxb = xa.nearrefup(1);
            const SparseVector<gentype> &rxc = xb.nearrefup(0);
            const SparseVector<gentype> &rxd = xc.nearrefup(0);

            const vecInfo &rxainfo = xainfo(-1,0);
            const vecInfo &rxbinfo = xainfo(-1,1);
            const vecInfo &rxcinfo = xbinfo(-1,0);
            const vecInfo &rxdinfo = xcinfo(-1,0);

            int ixa = ia;
            int ixb = UPNTVI(ia,1);
            int ixc = ib;
            int ixd = ic;

            int ixaset = iaset;
            int ixbset = iaset;
            int ixcset = ibset;
            int ixdset = icset;

            xKKK4(res,rxa,rxb,rxc,rxd,rxainfo,rxbinfo,rxcinfo,rxdinfo,bias,NULL,ixa,ixb,ixc,ixd,xdim,xconsist,assumreal,xresmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,ixaset,ixbset,ixcset,ixdset);
        }

        else
        {
            int ii;

            Vector<int> iv(aupm+bupm+cupm);
            Vector<const SparseVector<gentype> *> xxv(aupm+bupm+cupm);
            Vector<const vecInfo *> xxvinfo(aupm+bupm+cupm);
            Vector<int> ivset(aupm+bupm+cupm);

            if ( aupm )
            {
                for ( ii = 0 ; ii < aupm ; ii++ )
                {
                    iv("&",ii) = ii ? UPNTVI(ia,ii) : ia;
                    xxv("&",ii) = &(xa.nearrefup(ii));
                    xxvinfo("&",ii) = &(xainfo(-1,ii));
                    ivset("&",ii) = iaset;
                }
            }

            if ( bupm )
            {
                for ( ii = 0 ; ii < bupm ; ii++ )
                {
                    iv("&",ii+aupm) = ii ? UPNTVI(ib,ii) : ib;
                    xxv("&",ii+aupm) = &(xb.nearrefup(ii));
                    xxvinfo("&",ii+aupm) = &(xbinfo(-1,ii));
                    ivset("&",ii+aupm) = ibset;
                }
            }

            if ( cupm )
            {
                for ( ii = 0 ; ii < cupm ; ii++ )
                {
                    iv("&",ii+aupm+bupm) = ii ? UPNTVI(ic,ii) : ic;
                    xxv("&",ii+aupm+bupm) = &(xc.nearrefup(ii));
                    xxvinfo("&",ii+aupm+bupm) = &(xcinfo(-1,ii));
                    ivset("&",ii+aupm+bupm) = icset;
                }
            }

            xKKKm(aupm+bupm+cupm,res,xxv,xxvinfo,bias,iv,NULL,xdim,xconsist,assumreal,xresmode,mlid,NULL,justcalcip,&ivset);
        }

    return res;
}

template <class T>
T &MercerKernel::yyyKK4(T &res,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib, int ic, int id, 
                    int xdim, int xconsist, int xresmode, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int assumreal, int justcalcip) const
{
//    return xKKK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,assumreal,xresmode,mlid,0,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,iaset,ibset,icset,idset);
        int aupm = xa.nearupsize();
        int bupm = xb.nearupsize();
        int cupm = xc.nearupsize();
        int dupm = xd.nearupsize();

        if ( ( aupm == 1 ) && ( bupm == 1 ) && ( cupm == 1 ) && ( dupm == 1 ) )
        {
            xKKK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,assumreal,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,justcalcip,iaset,ibset,icset,idset);
        }

        else
        {
            int ii;

            Vector<int> iv(aupm+bupm+cupm+dupm);
            Vector<const SparseVector<gentype> *> xxv(aupm+bupm+cupm+dupm);
            Vector<const vecInfo *> xxvinfo(aupm+bupm+cupm+dupm);
            Vector<int> ivset(aupm+bupm+cupm+dupm);

            if ( aupm )
            {
                for ( ii = 0 ; ii < aupm ; ii++ )
                {
                    iv("&",ii) = ii ? UPNTVI(ia,ii) : ia;
                    xxv("&",ii) = &(xa.nearrefup(ii));
                    xxvinfo("&",ii) = &(xainfo(-1,ii));
                    ivset("&",ii) = iaset;
                }
            }

            if ( bupm )
            {
                for ( ii = 0 ; ii < bupm ; ii++ )
                {
                    iv("&",ii+aupm) = ii ? UPNTVI(ib,ii) : ib;
                    xxv("&",ii+aupm) = &(xb.nearrefup(ii));
                    xxvinfo("&",ii+aupm) = &(xbinfo(-1,ii));
                    ivset("&",ii+aupm) = ibset;
                }
            }

            if ( cupm )
            {
                for ( ii = 0 ; ii < cupm ; ii++ )
                {
                    iv("&",ii+aupm+bupm) = ii ? UPNTVI(ic,ii) : ic;
                    xxv("&",ii+aupm+bupm) = &(xc.nearrefup(ii));
                    xxvinfo("&",ii+aupm+bupm) = &(xcinfo(-1,ii));
                    ivset("&",ii+aupm+bupm) = icset;
                }
            }

            if ( dupm )
            {
                for ( ii = 0 ; ii < dupm ; ii++ )
                {
                    iv("&",ii+aupm+bupm+cupm) = ii ? UPNTVI(id,ii) : id;
                    xxv("&",ii+aupm+bupm+cupm) = &(xd.nearrefup(ii));
                    xxvinfo("&",ii+aupm+bupm+cupm) = &(xdinfo(-1,ii));
                    ivset("&",ii+aupm+bupm+cupm) = idset;
                }
            }

            xKKKm(aupm+bupm+cupm+dupm,res,xxv,xxvinfo,bias,iv,NULL,xdim,xconsist,assumreal,xresmode,mlid,NULL,justcalcip,&ivset);
        }

    return res;
}

template <class T>
T &MercerKernel::yyyKKm(int m, T &res,
                    Vector<const SparseVector<gentype> *> &x,
                    Vector<const vecInfo *> &xinfo,
                    const T &bias,
                    Vector<int> &i,
                    const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, 
                    const Matrix<double> *xy, const Vector<int> *iiset, int assumreal, int justcalcip) const
{
//    return xKKKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,xy,0,iset);
    int needpreproc = 0;
    int ii;

    Vector<int> altiset;

    if ( !iiset )
    {
        altiset.resize(m);
        altiset = zeroint();

        iiset = &altiset;
    }

    if ( m )
    {
        for ( ii = 0 ; ii < m ; ii++ )
        {
            const SparseVector<gentype> *xxi = x(ii);

            if ( (*xxi).nearupsize() > 1 )
            {
                needpreproc = 1;

                break;
            }
        }
    }

    if ( !needpreproc )
    {
        xKKKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,iiset);
    }

    else
    {
        Vector<int> j(i);
        Vector<const SparseVector<gentype> *> xx(x);
        Vector<const vecInfo *> xxinfo(xinfo);
        Vector<int> jset(*iiset);

        // Fill in any missing data vectors

        int ii,jj,u;

        for ( ii = 0 ; ii < m ; ii++ )
        {
            if ( ( u = (*(xx(ii))).nearupsize() ) > 1 )
            {
                const SparseVector<gentype> &xxv = (*(xx(ii)));
                const vecInfo &xxvinfo = (*(xxinfo(ii)));

                for ( jj = 0 ; jj < u ; jj++ )
                {
                    if ( jj )
                    {
                        m++;

                        j.add(ii+jj);
                        xx.add(ii+jj);
                        xxinfo.add(ii+jj);
                        jset.add(ii+jj);
                    }

                    j("&",ii+jj) = jj ? UPNTVI(j(ii+jj),jj) : j(ii+jj);
                    xx("&",ii+jj) =  &(xxv.nearrefup(jj));
                    xxinfo("&",ii+jj) = &(xxvinfo(ii+jj)(-1,jj));
                    jset("&",ii+jj) = jset(ii+jj);
                }
            }
        }

        xKKKm(xx.size(),res,xx,xxinfo,bias,j,NULL,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,&jset);
    }

    return res;
}







// When normalising kernels we need to take care that the result isn't 0/0 = nan.
// Hack: waves wand... 0/0 = 1.  Take that, maths.
// As an aside, this means we need to do normalisation as a *single* division, not
// multiple divisions - so K(x,y)/sqrt((K(x,x).K(y,y)) rather than 
// (K(x,y)/sqrt(K(x,x)))/sqrt(K(y,y)), because, by the above magic, in the worst 
// case scenario:
//
// K(x,y)/sqrt((K(x,x).K(y,y)) = 0/sqrt(0*0) = 0/0 = 1
// (K(x,y)/sqrt(K(x,x)))/sqrt(K(y,y)) = (0/sqrt(0))/sqrt(0) = (0/0)/0 = 1/0 = inf
//
// The former makes sense (up to sign) as a limit, the latter is just plain wrong.

template <class T>
T &safedivby(T &a, const T &b);
template <class T>
T &safedivby(T &a, const T &b)
{
    a /= b;

    if ( testisvnan(a) )
    {
        setident(a);
    }

    return a;
}










template <class T> T &MercerKernel::xKKK0(T &res, const T &bias, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int xresmode, int mlid, int justcalcip) const
{
    if ( numMulSplits() )
    {
        int q,r = 0;

        Vector<int> splitPoint(numMulSplits());

        if ( numMulSplits() )
        {
            for ( q = 0 ; q < numMulSplits() ; q++ )
            {
                if ( isMulSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        for ( q = 0 ; q <= numMulSplits() ; q++ )
        {
            int indstart = q ? splitPoint(q-1)+1 : 0;
            int indend   = ( q < numMulSplits() ) ? splitPoint(q) : size()-1;

            if ( !q )
            {
                xKK0(res,bias,pxyprod,xdim,xconsist,assumreal,xresmode,mlid,justcalcip,indstart,indend,calcnumSplits(indstart,indend));
            }

            else
            {
                T tempres;

                xKK0(tempres,bias,pxyprod,xdim,xconsist,assumreal,xresmode,mlid,justcalcip,indstart,indend,calcnumSplits(indstart,indend));

                res *= tempres;
            }
        }
    }

    else
    {
        xKK0(res,bias,pxyprod,xdim,xconsist,assumreal,xresmode,mlid,justcalcip,0,size()-1,numSplits());
    }

    return res;
}

template <class T> T &MercerKernel::xKKK1(T &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, const T &bias, const gentype **pxyprod, int ia, int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int iaset) const
{
    if ( numMulSplits() )
    {
        int q,r = 0;

        Vector<int> splitPoint(numMulSplits());

        if ( numMulSplits() )
        {
            for ( q = 0 ; q < numMulSplits() ; q++ )
            {
                if ( isMulSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        for ( q = 0 ; q <= numMulSplits() ; q++ )
        {
            int indstart = q ? splitPoint(q-1)+1 : 0;
            int indend   = ( q < numMulSplits() ) ? splitPoint(q) : size()-1;

            if ( !q )
            {
                xKK1(res,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,iaset,indstart,indend,calcnumSplits(indstart,indend));
            }

            else
            {
                T tempres;

                xKK1(tempres,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,iaset,indstart,indend,calcnumSplits(indstart,indend));

                res *= tempres;
            }
        }
    }

    else
    {
        xKK1(res,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,iaset,0,size()-1,numSplits());
    }

    return res;
}

template <class T> T &MercerKernel::xKKK2(T &res, 
                                          const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                                          const vecInfo &xainfo, const vecInfo &xbinfo, 
                                          const T &bias, const gentype **pxyprod, 
                                          int ia, int ib, 
                                          int xdim, int xconsist, int assumreal, int resmode, int mlid, 
                                          const double *xy00, const double *xy10, const double *xy11, int justcalcip, int iset, int jset) const
{
    if ( numMulSplits() )
    {
        int q,r = 0;

        Vector<int> splitPoint(numMulSplits());

        if ( numMulSplits() )
        {
            for ( q = 0 ; q < numMulSplits() ; q++ )
            {
                if ( isMulSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        for ( q = 0 ; q <= numMulSplits() ; q++ )
        {
            int indstart = q ? splitPoint(q-1)+1 : 0;
            int indend   = ( q < numMulSplits() ) ? splitPoint(q) : size()-1;

            if ( !q )
            {
                xKK2(res,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,justcalcip,iset,jset,indstart,indend,calcnumSplits(indstart,indend));
            }

            else
            {
                T tempres;

                xKK2(tempres,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,justcalcip,iset,jset,indstart,indend,calcnumSplits(indstart,indend));

                res *= tempres;
            }
        }
    }

    else
    {
        xKK2(res,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,justcalcip,iset,jset,0,size()-1,numSplits());
    }

    return res;
}

template <class T> T &MercerKernel::xKKK3(T &res, 
                                          const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                                          const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                                          const T &bias, 
                                          const gentype **pxyprod, 
                                          int ia, int ib, int ic, 
                                          int xdim, int xconsist, int assumreal, int xresmode, int mlid, 
                                          const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, 
                                          int justcalcip, int iaset, int ibset, int icset) const
{
    if ( numMulSplits() )
    {
        int q,r = 0;

        Vector<int> splitPoint(numMulSplits());

        if ( numMulSplits() )
        {
            for ( q = 0 ; q < numMulSplits() ; q++ )
            {
                if ( isMulSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        for ( q = 0 ; q <= numMulSplits() ; q++ )
        {
            int indstart = q ? splitPoint(q-1)+1 : 0;
            int indend   = ( q < numMulSplits() ) ? splitPoint(q) : size()-1;

            if ( !q )
            {
                xKK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,assumreal,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,justcalcip,iaset,ibset,icset,indstart,indend,calcnumSplits(indstart,indend));
            }

            else
            {
                T tempres;

                xKK3(tempres,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,assumreal,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,justcalcip,iaset,ibset,icset,indstart,indend,calcnumSplits(indstart,indend));

                res *= tempres;
            }
        }
    }

    else
    {
        xKK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,assumreal,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,justcalcip,iaset,ibset,icset,0,size()-1,numSplits());
    }

    return res;
}

template <class T> T &MercerKernel::xKKK4(T &res, 
                                          const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, 
                                          const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, 
                                          const T &bias, const gentype **pxyprod, 
                                          int ia, int ib, int ic, int id, 
                                          int xdim, int xconsist, int assumreal, int xresmode, int mlid, 
                                          const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int justcalcip, int iaset, int ibset, int icset, int idset) const
{
    if ( numMulSplits() )
    {
        int q,r = 0;

        Vector<int> splitPoint(numMulSplits());

        if ( numMulSplits() )
        {
            for ( q = 0 ; q < numMulSplits() ; q++ )
            {
                if ( isMulSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        for ( q = 0 ; q <= numMulSplits() ; q++ )
        {
            int indstart = q ? splitPoint(q-1)+1 : 0;
            int indend   = ( q < numMulSplits() ) ? splitPoint(q) : size()-1;

            if ( !q )
            {
                xKK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,assumreal,xresmode,mlid,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,iaset,ibset,icset,idset,indstart,indend,calcnumSplits(indstart,indend));
            }

            else
            {
                T tempres;

                xKK4(tempres,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,assumreal,xresmode,mlid,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,iaset,ibset,icset,idset,indstart,indend,calcnumSplits(indstart,indend));

                res *= tempres;
            }
        }
    }

    else
    {
        xKK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,assumreal,xresmode,mlid,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,iaset,ibset,icset,idset,0,size()-1,numSplits());
    }

    return res;
}

template <class T> T &MercerKernel::xKKKm(int m, T &res, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const T &bias, Vector<int> &i, const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, const Matrix<double> *xy, int justcalcip, const Vector<int> *iset) const
{
    if ( numMulSplits() )
    {
        int q,r = 0;

        Vector<int> splitPoint(numMulSplits());

        if ( numMulSplits() )
        {
            for ( q = 0 ; q < numMulSplits() ; q++ )
            {
                if ( isMulSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        for ( q = 0 ; q <= numMulSplits() ; q++ )
        {
            int indstart = q ? splitPoint(q-1)+1 : 0;
            int indend   = ( q < numMulSplits() ) ? splitPoint(q) : size()-1;

            if ( !q )
            {
                xKKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,iset,indstart,indend,calcnumSplits(indstart,indend));
            }

            else
            {
                T tempres;

                xKKm(m,tempres,x,xinfo,bias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,iset,indstart,indend,calcnumSplits(indstart,indend));

                res *= tempres;
            }
        }
    }

    else
    {
        xKKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,iset,0,size()-1,numSplits());
    }

    return res;
}









//phantomx
template <class T>
T &MercerKernel::xKK0(T &res,
                     const T &bias,
                     const gentype **pxyprod,
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, int justcalcip, int indstart, int indend, int ns) const
{
    T logres; logres = 0.0;
    int logresvalid = 0;
    //int ns = numSplits();

    if ( isfullnorm && !ns && !justcalcip )
    {
        res = 1.0;
    }

    else if ( !isfullnorm && ns && !justcalcip )
    {
        res = 1.0; // Design decision: empty product evaluates to 1
    }

    else if ( isfullnorm && ns && !justcalcip )
    {
        res = 1.0; // Design decision: empty product evaluates to 1
    }

    else
    {
        KK0(res,logres,logresvalid,bias,pxyprod,xdim,xconsist,assumreal,resmode,mlid,justcalcip,indstart,indend);
    }

    return res;
}

//phantomx
template <class T>
T &MercerKernel::xKK1(T &res,
                     const SparseVector<gentype> &xa, 
                     const vecInfo &xainfo, 
                     const T &bias,
                     const gentype **pxyprod,
                     int ia,  
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, 
                     const double *xy, int justcalcip, int iset, int indstart, int indend, int ns) const
{
    T logres; logres = 0.0;
    int logresvalid = 0;
    //int ns = numSplits();

    if ( isfullnorm && !ns && !justcalcip )
    {
        res = 1.0;
    }

    else if ( !isfullnorm && ns && !justcalcip )
    {
        int indstarta = indstart;
        int indenda   = indend;

        if ( ns )
        {
            int q;

            for ( q = indstart ; q < indend ; q++ )
            {
                if ( isSplit(q) )
                {
                    indenda = q;
                    break;
                }
            }
        }

        KK1(res,logres,logresvalid,xa,xainfo,bias,NULL,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstarta,indenda,iset);
    }

    else if ( isfullnorm && ns && !justcalcip )
    {
        res = 1.0;
    }

    else
    {
        KK1(res,logres,logresvalid,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,indstart,indend,iset);
    }

    return res;
}

//phantomx
template <class T>
T &MercerKernel::xKK2(T &res,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                     const vecInfo &xainfo, const vecInfo &xbinfo,
                     const T &bias,
                     const gentype **pxyprod,
                     int ia, int ib,
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, 
                     const double *xy00, const double *xy10, const double *xy11, int justcalcip, int iset, int jset, int indstart, int indend, int ns) const
{
    T logres; logres = 0.0;
    int logresvalid = 0;
    //int ns = numSplits();
    int q,r = 0;
    int z = 0;

    if ( isfullnorm && !ns && !justcalcip )
    {
        // 1:1, 2:2

        if ( xainfo.xusize() == 1 )
        {
            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid = 0;
            int logtmbvalid = 0;

            KK2(res,logres,logresvalid,xa,xb,xainfo,xbinfo,bias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,justcalcip,indstart,indend,iset,jset);

            KK2(tma,logtma,logtmavalid,xa,xa,xainfo,xainfo,bias,NULL,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstart,indend,iset,iset);
            KK2(tmb,logtmb,logtmbvalid,xb,xb,xbinfo,xbinfo,bias,NULL,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstart,indend,jset,jset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); // res /= tma;
            }
        }

        else
        {
            res = 1.0;
        }
    }

    else if ( !isfullnorm && ns && !justcalcip )
    {
        T tempres; tempres = 0.0;

        Vector<int> splitPoint(ns);

        if ( ns )
        {
            for ( q = indstart ; q < indend ; q++ )
            {
                if ( isSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        int indstarta = indstart;
        int indenda   = indend;

        int indstartb = indstart;
        int indendb   = indend;

        if ( ns == 1 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = indend;
        }

        else if ( ns >= 2 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = splitPoint(1);
        }

        // 1:1, 2:2

        if ( xainfo.xusize() == 1 )
        {
            KK2(res,logres,logresvalid,xa,xb,xainfo,xbinfo,bias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iset,jset);
        }

        else
        {
            // xainfo.xusize() == 2

            KK1(    res,logres,logresvalid,xa,xainfo,bias,NULL,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstarta,indenda,iset);
            KK1(tempres,logres,logresvalid,xb,xbinfo,bias,NULL,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,jset);

            res *= tempres;
        }
    }

    else if ( isfullnorm && ns && !justcalcip )
    {
        Vector<int> splitPoint(ns);

        if ( ns )
        {
            for ( q = indstart ; q < indend ; q++ )
            {
                if ( isSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        int indstarta = indstart;
        int indenda   = indend;
//
//        int indstartb = indstart;
//        int indendb   = indend;
//
        if ( ns == 1 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);
//
//            indstartb = splitPoint(z)+1;
//            indendb   = indend;
        }

        else if ( ns >= 2 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);
//
//            indstartb = splitPoint(z)+1;
//            indendb   = splitPoint(1);
        }

        // 1:1, 2:2

        if ( xainfo.xusize() == 1 )
        {
            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid = 0;
            int logtmbvalid = 0;

            KK2(res,logres,logresvalid,xa,xb,xainfo,xbinfo,bias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iset,jset);

            KK2(tma,logtma,logtmavalid,xa,xa,xainfo,xainfo,bias,NULL,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iset,iset);
            KK2(tmb,logtmb,logtmbvalid,xb,xb,xbinfo,xbinfo,bias,NULL,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,jset,jset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); // res /= tma;
            }
        }

        else
        {
            res = 1.0;
        }
    }

    else
    {
        KK2(res,logres,logresvalid,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,justcalcip,indstart,indend,iset,jset);
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return res;
}

//phantomx
template <class T>
T &MercerKernel::xKK3(T &res,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                     const T &bias,
                     const gentype **pxyprod,
                     int ia, int ib, int ic, 
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, 
                     const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int justcalcip, int iaset, int ibset, int icset, int indstart, int indend, int ns) const
{
    T logres; logres = 0.0;
    T tempres; tempres = 0.0;
    T xtempres; xtempres = 0.0;
    int logresvalid = 0;
    //int ns = numSplits();
    int q,r = 0;
    int z = 0;

    if ( isfullnorm && !ns && !justcalcip )
    {
        // 1:1:1, 
        // 2:2:1, 1:2:2
        // 3:3:3

        if ( ( xainfo.xusize() == 1 ) && ( xbinfo.xusize() == 1 ) )
        {
            T tma;
            T tmb;
            T tmc;

            T logtma;
            T logtmb;
            T logtmc;

            int logtmavalid;
            int logtmbvalid;
            int logtmcvalid;

            KK3(res,logres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,NULL,justcalcip,indstart,indend,iaset,ibset,icset);

            KK3(tma,logtma,logtmavalid,xa,xa,xa,xainfo,xainfo,xainfo,bias,NULL,ia,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,iaset,iaset);
            KK3(tmb,logtmb,logtmbvalid,xb,xb,xb,xbinfo,xbinfo,xbinfo,bias,NULL,ib,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,ibset,ibset,ibset);
            KK3(tmc,logtmc,logtmcvalid,xc,xc,xc,xcinfo,xcinfo,xcinfo,bias,NULL,ic,ic,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,icset,icset,icset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid )
            {
                logtma /= 3.0;
                logtmb /= 3.0;
                logtmc /= 3.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;

                OP_exp(res);
            }

            else
            {
                T oneonm; oneonm = 1.0/3.0;

                tma *= tmb;
                tma *= tmc;
                tma = pow(tma,oneonm);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( xainfo.xusize() == 2 )
        {
            // This is actually a 2-kernel evaluation in disguise!

            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid;
            int logtmbvalid;

            KK3(res,logres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,NULL,justcalcip,indstart,indend,iaset,ibset,icset);

            KK4(tma,logtma,logtmavalid,xa,xb,xa,xb,xainfo,xbinfo,xainfo,xbinfo,bias,NULL,ia,ib,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend,iaset,ibset,iaset,ibset);
            KK2(tmb,logtmb,logtmbvalid,xc,xc,xcinfo,xcinfo,bias,NULL,ic,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstart,indend,icset,icset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( xbinfo.xusize() == 2 )
        {
            // This is actually a 2-kernel evaluation in disguise!

            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid;
            int logtmbvalid;

            KK3(res,logres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,NULL,justcalcip,indstart,indend,iaset,ibset,icset);

            KK2(tma,logtma,logtmavalid,xa,xa,xainfo,xainfo,bias,NULL,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,iaset);
            KK4(tmb,logtmb,logtmbvalid,xb,xc,xb,xc,xbinfo,xcinfo,xbinfo,xcinfo,bias,NULL,ib,ic,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend,ibset,icset,ibset,icset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else
        {
            res = 1.0;
        }
    }

    else if ( !isfullnorm && ns && !justcalcip )
    {
        Vector<int> splitPoint(ns);

        if ( ns )
        {
            for ( q = indstart ; q < indend ; q++ )
            {
                if ( isSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        int indstarta = indstart;
        int indenda   = indend;

        int indstartb = indstart;
        int indendb   = indend;

        int indstartc = indstart;
        int indendc   = indend;

        if ( ns == 1 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = indend;

            indstartc = indstart;
            indendc   = splitPoint(z);
        }

        else if ( ns == 2 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = splitPoint(1);

            indstartc = splitPoint(1)+1;
            indendc   = indend;
        }

        else if ( ns >= 3 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = splitPoint(1);

            indstartc = splitPoint(1)+1;
            indendc   = splitPoint(2);
        }

        // 1:1:1, 
        // 2:2:1, 1:2:2
        // 3:3:3

        if ( ( xainfo.xusize() == 1 ) && ( xbinfo.xusize() == 1 ) )
        {
            KK3(res,logres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset,icset);
        }

        else if ( xainfo.xusize() == 2 )
        {
            KK2(    res,logres,logresvalid,xa,xc,xainfo,xcinfo,bias,NULL,ia,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,icset);
            KK1(tempres,logres,logresvalid,xb,xbinfo,bias,NULL,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,ibset);

            res *= tempres;
        }

        else if ( xbinfo.xusize() == 2 )
        {
            KK2(    res,logres,logresvalid,xa,xb,xainfo,xbinfo,bias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset);
            KK1(tempres,logres,logresvalid,xc,xcinfo,bias,NULL,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,icset);

            res *= tempres;
        }

        else
        {
            // xainfo.usize() == 3

            KK1(     res,logres,logresvalid,xa,xainfo,bias,NULL,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstarta,indenda,iaset);
            KK1( tempres,logres,logresvalid,xb,xbinfo,bias,NULL,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,ibset);
            KK1(xtempres,logres,logresvalid,xc,xcinfo,bias,NULL,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartc,indendc,icset);

            res *=  tempres;
            res *= xtempres;
        }
    }

    else if ( isfullnorm && ns && !justcalcip )
    {
        Vector<int> splitPoint(ns);

        if ( ns )
        {
            for ( q = indstart ; q < indend ; q++ )
            {
                if ( isSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }


        int indstarta = indstart;
        int indenda   = indend;

//        int indstartb = indstart;
//        int indendb   = indend;
//
//        int indstartc = indstart;
//        int indendc   = indend;
//
        if ( ns == 1 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);
//
//            indstartb = splitPoint(z)+1;
//            indendb   = indend;
//
//            indstartc = indstart;
//            indendc   = splitPoint(z);
        }

        else if ( ns == 2 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);
//
//            indstartb = splitPoint(z)+1;
//            indendb   = splitPoint(1);
//
//            indstartc = splitPoint(1)+1;
//            indendc   = indend;
        }

        else if ( ns >= 3 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);
//
//            indstartb = splitPoint(z)+1;
//            indendb   = splitPoint(1);
//
//            indstartc = splitPoint(1)+1;
//            indendc   = splitPoint(2);
        }

        // 1:1:1, 
        // 2:2:1, 1:2:2
        // 3:3:3

        if ( ( xainfo.xusize() == 1 ) && ( xbinfo.xusize() == 1 ) )
        {
            T tma;
            T tmb;
            T tmc;

            T logtma;
            T logtmb;
            T logtmc;

            int logtmavalid = 0;
            int logtmbvalid = 0;
            int logtmcvalid = 0;

            KK3(res,logres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset,icset);

            KK3(tma,logtma,logtmavalid,xa,xa,xa,xainfo,xainfo,xainfo,bias,NULL,ia,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,iaset,iaset);
            KK3(tmb,logtmb,logtmbvalid,xb,xb,xb,xbinfo,xbinfo,xbinfo,bias,NULL,ib,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,ibset,ibset,ibset);
            KK3(tmc,logtmc,logtmcvalid,xc,xc,xc,xcinfo,xcinfo,xcinfo,bias,NULL,ic,ic,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,icset,icset,icset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid )
            {
                logtma /= 3.0;
                logtmb /= 3.0;
                logtmc /= 3.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                tma *= tmc;

                T sf; sf = 1.0/3.0;

                safedivby(res,pow(tma,sf)); //res /= pow(tma,sf);
            }
        }

        else if ( xainfo.xusize() == 2 )
        {
            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid = 0;
            int logtmbvalid = 0;

            KK2(res,logres,logresvalid,xa,xc,xainfo,xcinfo,bias,NULL,ia,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,icset);

            KK2(tma,logtma,logtmavalid,xa,xa,xainfo,xainfo,bias,NULL,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,iaset);
            KK2(tmb,logtmb,logtmbvalid,xc,xc,xcinfo,xcinfo,bias,NULL,ic,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,icset,icset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( xbinfo.xusize() == 2 )
        {
            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid = 0;
            int logtmbvalid = 0;

            KK2(res,logres,logresvalid,xa,xb,xainfo,xbinfo,bias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset);

            KK2(tma,logtma,logtmavalid,xa,xa,xainfo,xainfo,bias,NULL,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,iaset);
            KK2(tmb,logtmb,logtmbvalid,xb,xb,xbinfo,xbinfo,bias,NULL,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,ibset,ibset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else
        {
            res = 1.0;
        }
    }

    else
    {
        KK3(res,logres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,NULL,justcalcip,indstart,indend,iaset,ibset,icset);
    }




    return res;
}

//phantomx
template <class T>
T &MercerKernel::xKK4(T &res,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                     const T &bias,
                     const gentype **pxyprod,
                     int ia, int ib, int ic, int id, 
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, int justcalcip, 
                     const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int iaset, int ibset, int icset, int idset, int indstart, int indend, int ns) const
{
    T logres; logres = 0.0;
    T atempres; atempres = 0.0;
    T btempres; btempres = 0.0;
    T ctempres; ctempres = 0.0;
    int logresvalid = 0;
    //int ns = numSplits();
    int q,r = 0;
    int z = 0;

    if ( isfullnorm && !ns && !justcalcip )
    {
        // 1:1:1:1
        // 2:2:1:1, 1:2:2:1, 1:1:2:2
        // 2:2:2:2
        // 3:3:3:1, 1:3:3:3
        // 4:4:4:4

        if ( ( xainfo.xusize() == 1 ) && ( xbinfo.xusize() == 1 ) && ( xcinfo.xusize() == 1 ) )
        {
            T tma;
            T tmb;
            T tmc;
            T tmd;

            T logtma;
            T logtmb;
            T logtmc;
            T logtmd;

            int logtmavalid;
            int logtmbvalid;
            int logtmcvalid;
            int logtmdvalid;

            KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,indstart,indend,iaset,ibset,icset,idset);

            KK4(tma,logtma,logtmavalid,xa,xa,xa,xa,xainfo,xainfo,xainfo,xainfo,bias,NULL,ia,ia,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy00,xy00,xy00,xy00,xy00,xy00,xy00,xy00,xy00,indstart,indend,iaset,iaset,iaset,iaset);
            KK4(tmb,logtmb,logtmbvalid,xb,xb,xb,xb,xbinfo,xbinfo,xbinfo,xbinfo,bias,NULL,ib,ib,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy11,xy11,xy11,xy11,xy11,xy11,xy11,xy11,xy11,xy11,indstart,indend,ibset,ibset,ibset,ibset);
            KK4(tmc,logtmc,logtmcvalid,xc,xc,xc,xc,xcinfo,xcinfo,xcinfo,xcinfo,bias,NULL,ic,ic,ic,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy22,xy22,xy22,xy22,xy22,xy22,xy22,xy22,xy22,xy22,indstart,indend,icset,icset,icset,icset);
            KK4(tmd,logtmd,logtmdvalid,xd,xd,xd,xd,xdinfo,xdinfo,xdinfo,xdinfo,bias,NULL,id,id,id,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy33,xy33,xy33,xy33,xy33,xy33,xy33,xy33,xy33,xy33,indstart,indend,idset,idset,idset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid && logtmdvalid )
            {
                logtma /= 4.0;
                logtmb /= 4.0;
                logtmc /= 4.0;
                logtmd /= 4.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;
                res -= logtmd;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                tma *= tmc;
                tma *= tmd;
                OP_sqrt(tma);
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( ( xainfo.xusize() == 2 ) && ( xcinfo.xusize() == 1 ) )
        {
            T tma;
            T tmb;
            T tmc;

            T logtma;
            T logtmb;
            T logtmc;

            int logtmavalid;
            int logtmbvalid;
            int logtmcvalid;

            KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,indstart,indend,iaset,ibset,icset,idset);

            KK6(tma,logtma,logtmavalid,xa,xb,xa,xb,xa,xb,xainfo,xbinfo,xainfo,xbinfo,xainfo,xbinfo,bias,NULL,ia,ib,ia,ib,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstart,indend,iaset,ibset,iaset,ibset,iaset,ibset);
            KK3(tmb,logtmb,logtmbvalid,xc,xc,xc,xcinfo,xcinfo,xcinfo,bias,NULL,ic,ic,ic,xdim,xconsist,assumreal,mlid,resmode,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,icset,icset,icset);
            KK3(tmc,logtmc,logtmcvalid,xd,xd,xd,xdinfo,xdinfo,xdinfo,bias,NULL,id,id,id,xdim,xconsist,assumreal,mlid,resmode,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,idset,idset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid )
            {
                logtma /= 3.0;
                logtmb /= 3.0;
                logtmc /= 3.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;

                OP_exp(res);
            }

            else
            {
                T oneonm; oneonm = 1.0/3.0;

                tma *= tmb;
                tma *= tmc;
                tma = pow(tma,oneonm);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( ( xainfo.xusize() == 1 ) && ( xbinfo.xusize() == 2 ) )
        {
            T tma;
            T tmb;
            T tmc;

            T logtma;
            T logtmb;
            T logtmc;

            int logtmavalid;
            int logtmbvalid;
            int logtmcvalid;

            KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,indstart,indend,iaset,ibset,icset,idset);

            KK3(tma,logtma,logtmavalid,xa,xa,xa,xainfo,xainfo,xainfo,bias,NULL,ia,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,iaset,iaset);
            KK6(tmb,logtmb,logtmbvalid,xb,xc,xb,xc,xb,xc,xbinfo,xcinfo,xbinfo,xcinfo,xbinfo,xcinfo,bias,NULL,ib,ic,ib,ic,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstart,indend,ibset,icset,ibset,icset,ibset,icset);
            KK3(tmc,logtmc,logtmcvalid,xd,xd,xd,xdinfo,xdinfo,xdinfo,bias,NULL,id,id,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,idset,idset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid )
            {
                logtma /= 3.0;
                logtmb /= 3.0;
                logtmc /= 3.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;

                OP_exp(res);
            }

            else
            {
                T oneonm; oneonm = 1.0/3.0;

                tma *= tmb;
                tma *= tmc;
                tma = pow(tma,oneonm);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( ( xainfo.xusize() == 1 ) && ( xcinfo.xusize() == 2 ) )
        {
            T tma;
            T tmb;
            T tmc;

            T logtma;
            T logtmb;
            T logtmc;

            int logtmavalid;
            int logtmbvalid;
            int logtmcvalid;

            KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,indstart,indend,iaset,ibset,icset,idset);

            KK3(tma,logtma,logtmavalid,xa,xa,xa,xainfo,xainfo,xainfo,bias,NULL,ia,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,iaset,iaset);
            KK3(tmb,logtmb,logtmbvalid,xb,xb,xb,xbinfo,xbinfo,xbinfo,bias,NULL,ib,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,ibset,ibset,ibset);
            KK6(tmc,logtmc,logtmcvalid,xc,xd,xc,xd,xc,xd,xcinfo,xdinfo,xcinfo,xdinfo,xcinfo,xdinfo,bias,NULL,ic,id,ic,id,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstart,indend,icset,idset,icset,idset,icset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid )
            {
                logtma /= 3.0;
                logtmb /= 3.0;
                logtmc /= 3.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;

                OP_exp(res);
            }

            else
            {
                T oneonm; oneonm = 1.0/3.0;

                tma *= tmb;
                tma *= tmc;
                tma = pow(tma,oneonm);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( ( xainfo.xusize() == 2 ) && ( xcinfo.xusize() == 2 ) )
        {
            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid;
            int logtmbvalid;

            KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,indstart,indend,iaset,ibset,icset,idset);

            KK4(tma,logtma,logtmavalid,xa,xb,xa,xb,xainfo,xbinfo,xainfo,xbinfo,bias,NULL,ia,ib,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy10,xy11,xy00,xy10,xy00,xy10,xy11,xy10,xy11,indstart,indend,iaset,ibset,iaset,ibset);
            KK4(tmb,logtmb,logtmbvalid,xc,xd,xc,xd,xcinfo,xdinfo,xcinfo,xdinfo,bias,NULL,ic,id,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy22,xy32,xy33,xy22,xy32,xy22,xy32,xy33,xy32,xy33,indstart,indend,icset,idset,icset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( xainfo.xusize() == 3 )
        {
            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid;
            int logtmbvalid;

            KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,indstart,indend,iaset,ibset,icset,idset);

            KK6(tma,logtma,logtmavalid,xa,xb,xc,xa,xb,xc,xainfo,xbinfo,xcinfo,xainfo,xbinfo,xcinfo,bias,NULL,ia,ib,ic,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstart,indend,iaset,ibset,icset,iaset,ibset,icset);
            KK2(tmb,logtmb,logtmbvalid,xd,xd,xdinfo,xdinfo,bias,NULL,id,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstart,indend,idset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( xbinfo.xusize() == 3 )
        {
            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid;
            int logtmbvalid;

            KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,indstart,indend,iaset,ibset,icset,idset);

            KK2(tmb,logtma,logtmavalid,xa,xa,xainfo,xainfo,bias,NULL,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,iaset);
            KK6(tma,logtmb,logtmbvalid,xb,xc,xd,xb,xc,xd,xbinfo,xcinfo,xdinfo,xbinfo,xcinfo,xdinfo,bias,NULL,ib,ic,id,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstart,indend,ibset,icset,idset,ibset,icset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else
        {
            res = 1.0;
        }
    }

    else if ( !isfullnorm && ns && !justcalcip )
    {
        Vector<int> splitPoint(ns);

        if ( ns )
        {
            for ( q = indstart ; q < indend ; q++ )
            {
                if ( isSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        int indstarta = indstart;
        int indenda   = indend;

        int indstartb = indstart;
        int indendb   = indend;

        int indstartc = indstart;
        int indendc   = indend;

        int indstartd = indstart;
        int indendd   = indend;

        if ( ns == 1 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = indend;

            indstartc = indstart;
            indendc   = splitPoint(z);

            indstartd = splitPoint(z)+1;
            indendd   = indend;
        }

        else if ( ns == 2 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = splitPoint(1);

            indstartc = splitPoint(1)+1;
            indendc   = indend;

            indstartd = indstart;
            indendd   = splitPoint(z);
        }

        else if ( ns == 3 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = splitPoint(1);

            indstartc = splitPoint(1)+1;
            indendc   = splitPoint(2);

            indstartd = splitPoint(2)+1;
            indendd   = indend;
        }

        else if ( ns >= 4 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = splitPoint(1);

            indstartc = splitPoint(1)+1;
            indendc   = splitPoint(2);

            indstartd = splitPoint(2)+1;
            indendd   = splitPoint(3);
        }

        // 1:1:1:1
        // 2:2:1:1, 1:2:2:1, 1:1:2:2
        // 2:2:2:2
        // 3:3:3:1, 1:3:3:3
        // 4:4:4:4

        if ( ( xainfo.xusize() == 1 ) && ( xbinfo.xusize() == 1 ) && ( xcinfo.xusize() == 1 ) )
        {
            KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstarta,indenda,iaset,ibset,icset,idset);
        }

        else if ( ( xainfo.xusize() == 2 ) && ( xcinfo.xusize() == 1 ) )
        {
            KK3(     res,logres,logresvalid,xa,xc,xd,xainfo,xcinfo,xdinfo,bias,NULL,ia,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,icset,idset);
            KK1(atempres,logres,logresvalid,xb,xbinfo,bias,NULL,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,ibset);

            res *= atempres;
        }

        else if ( ( xainfo.xusize() == 1 ) && ( xbinfo.xusize() == 2 ) )
        {
            KK3(     res,logres,logresvalid,xa,xb,xd,xainfo,xbinfo,xdinfo,bias,NULL,ia,ib,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset,idset);
            KK1(atempres,logres,logresvalid,xc,xcinfo,bias,NULL,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,icset);

            res *= atempres;
        }

        else if ( ( xainfo.xusize() == 1 ) && ( xcinfo.xusize() == 2 ) )
        {
            KK3(     res,logres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset,icset);
            KK1(atempres,logres,logresvalid,xd,xdinfo,bias,NULL,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,idset);

            res *= atempres;
        }

        else if ( ( xainfo.xusize() == 2 ) && ( xcinfo.xusize() == 2 ) )
        {
            KK2(     res,logres,logresvalid,xa,xc,xainfo,xcinfo,bias,NULL,ia,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,icset);
            KK2(atempres,logres,logresvalid,xb,xd,xbinfo,xdinfo,bias,NULL,ib,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstartb,indendb,ibset,idset);

            res *= atempres;
        }

        else if ( xainfo.xusize() == 3 )
        {
            KK2(     res,logres,logresvalid,xa,xd,xainfo,xdinfo,bias,NULL,ia,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,idset);
            KK1(atempres,logres,logresvalid,xb,xbinfo,bias,NULL,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,ibset);
            KK1(btempres,logres,logresvalid,xc,xcinfo,bias,NULL,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartc,indendc,icset);

            res *= atempres;
            res *= btempres;
        }

        else if ( xbinfo.xusize() == 3 )
        {
            KK2(     res,logres,logresvalid,xa,xb,xainfo,xbinfo,bias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset);
            KK1(atempres,logres,logresvalid,xc,xcinfo,bias,NULL,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,icset);
            KK1(btempres,logres,logresvalid,xd,xdinfo,bias,NULL,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartc,indendc,idset);

            res *= atempres;
            res *= btempres;
        }

        else
        {
            KK1(     res,logres,logresvalid,xa,xainfo,bias,NULL,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstarta,indenda,iaset);
            KK1(atempres,logres,logresvalid,xb,xbinfo,bias,NULL,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartb,indendb,ibset);
            KK1(btempres,logres,logresvalid,xc,xcinfo,bias,NULL,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartc,indendc,icset);
            KK1(ctempres,logres,logresvalid,xd,xdinfo,bias,NULL,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstartd,indendd,idset);

            res *= atempres;
            res *= btempres;
            res *= ctempres;
        }
    }

    else if ( isfullnorm && ns && !justcalcip )
    {
        Vector<int> splitPoint(ns);

        if ( ns )
        {
            for ( q = indstart ; q < indend ; q++ )
            {
                if ( isSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        int indstarta = indstart;
        int indenda   = indend;

        int indstartb = indstart;
        int indendb   = indend;

//        int indstartc = indstart;
//        int indendc   = indend;
//
//        int indstartd = indstart;
//        int indendd   = indend;

        if ( ns == 1 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = indend;

//            indstartc = indstart;
//            indendc   = splitPoint(z);
//
//            indstartd = splitPoint(z)+1;
//            indendd   = indend;
        }

        else if ( ns == 2 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = splitPoint(1);

//            indstartc = splitPoint(1)+1;
//            indendc   = indend;
//
//            indstartd = indstart;
//            indendd   = splitPoint(z);
        }

        else if ( ns == 3 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = splitPoint(1);

//            indstartc = splitPoint(1)+1;
//            indendc   = splitPoint(2);
//
//            indstartd = splitPoint(2)+1;
//            indendd   = indend;
        }

        else if ( ns >= 4 )
        {
            indstarta = indstart;
            indenda   = splitPoint(z);

            indstartb = splitPoint(z)+1;
            indendb   = splitPoint(1);

//            indstartc = splitPoint(1)+1;
//            indendc   = splitPoint(2);
//
//            indstartd = splitPoint(2)+1;
//            indendd   = splitPoint(3);
        }

        // 1:1:1:1
        // 2:2:1:1, 1:2:2:1, 1:1:2:2
        // 2:2:2:2
        // 3:3:3:1, 1:3:3:3
        // 4:4:4:4

        if ( ( xainfo.xusize() == 1 ) && ( xbinfo.xusize() == 1 ) && ( xcinfo.xusize() == 1 ) )
        {
            T tma;
            T tmb;
            T tmc;
            T tmd;

            T logtma;
            T logtmb;
            T logtmc;
            T logtmd;

            int logtmavalid = 0;
            int logtmbvalid = 0;
            int logtmcvalid = 0;
            int logtmdvalid = 0;

            KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstarta,indenda,iaset,ibset,icset,idset);

            KK4(tma,logtma,logtmavalid,xa,xa,xa,xa,xainfo,xainfo,xainfo,xainfo,bias,NULL,ia,ia,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstarta,indenda,iaset,iaset,iaset,iaset);
            KK4(tmb,logtmb,logtmbvalid,xb,xb,xb,xb,xbinfo,xbinfo,xbinfo,xbinfo,bias,NULL,ib,ib,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstarta,indenda,ibset,ibset,ibset,ibset);
            KK4(tmc,logtmc,logtmcvalid,xc,xc,xc,xc,xcinfo,xcinfo,xcinfo,xcinfo,bias,NULL,ic,ic,ic,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstarta,indenda,icset,icset,icset,icset);
            KK4(tmd,logtmd,logtmdvalid,xd,xd,xd,xd,xdinfo,xdinfo,xdinfo,xdinfo,bias,NULL,id,id,id,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstarta,indenda,idset,idset,idset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid && logtmdvalid )
            {
                logtma /= 4.0;
                logtmb /= 4.0;
                logtmc /= 4.0;
                logtmd /= 4.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;
                res -= logtmd;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                tma *= tmc;
                tma *= tmd;

                T sf; sf = 1.0/4.0;

                safedivby(res,pow(tma,sf)); //res /= pow(tma,sf);
            }
        }

        else if ( ( xainfo.xusize() == 2 ) && ( xcinfo.xusize() == 1 ) )
        {
            T tma;
            T tmb;
            T tmc;

            T logtma;
            T logtmb;
            T logtmc;

            int logtmavalid = 0;
            int logtmbvalid = 0;
            int logtmcvalid = 0;

            KK3(res,logres,logresvalid,xa,xc,xd,xainfo,xcinfo,xdinfo,bias,NULL,ia,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,icset,idset);

            KK3(tma,logtma,logtmavalid,xa,xa,xa,xainfo,xainfo,xainfo,bias,NULL,ia,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,iaset,iaset);
            KK3(tmb,logtmb,logtmbvalid,xc,xc,xc,xcinfo,xcinfo,xcinfo,bias,NULL,ic,ic,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,icset,icset,icset);
            KK3(tmc,logtmc,logtmcvalid,xd,xd,xd,xdinfo,xdinfo,xdinfo,bias,NULL,id,id,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,idset,idset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid )
            {
                logtma /= 3.0;
                logtmb /= 3.0;
                logtmc /= 3.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                tma *= tmc;

                T sf; sf = 1.0/3.0;

                safedivby(res,pow(tma,sf)); //res /= pow(tma,sf);
            }
        }

        else if ( ( xainfo.xusize() == 1 ) && ( xbinfo.xusize() == 2 ) )
        {
            T tma;
            T tmb;
            T tmc;

            T logtma;
            T logtmb;
            T logtmc;

            int logtmavalid = 0;
            int logtmbvalid = 0;
            int logtmcvalid = 0;

            KK3(res,logres,logresvalid,xa,xb,xd,xainfo,xbinfo,xdinfo,bias,NULL,ia,ib,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset,idset);

            KK3(tma,logtma,logtmavalid,xa,xa,xa,xainfo,xainfo,xainfo,bias,NULL,ia,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,iaset,iaset);
            KK3(tmb,logtmb,logtmbvalid,xb,xb,xb,xbinfo,xbinfo,xbinfo,bias,NULL,ib,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,ibset,ibset,ibset);
            KK3(tmc,logtmc,logtmcvalid,xd,xd,xd,xdinfo,xdinfo,xdinfo,bias,NULL,id,id,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,idset,idset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid )
            {
                logtma /= 3.0;
                logtmb /= 3.0;
                logtmc /= 3.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                tma *= tmc;

                T sf; sf = 1.0/3.0;

                safedivby(res,pow(tma,sf)); //res /= pow(tma,sf);
            }
        }

        else if ( ( xainfo.xusize() == 1 ) && ( xcinfo.xusize() == 2 ) )
        {
            T tma;
            T tmb;
            T tmc;

            T logtma;
            T logtmb;
            T logtmc;

            int logtmavalid = 0;
            int logtmbvalid = 0;
            int logtmcvalid = 0;

            KK3(res,logres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset,icset);

            KK3(tma,logtma,logtmavalid,xa,xa,xa,xainfo,xainfo,xainfo,bias,NULL,ia,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,iaset,iaset);
            KK3(tmb,logtmb,logtmbvalid,xb,xb,xb,xbinfo,xbinfo,xbinfo,bias,NULL,ib,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,ibset,ibset,ibset);
            KK3(tmc,logtmc,logtmcvalid,xc,xc,xc,xcinfo,xcinfo,xcinfo,bias,NULL,ic,ic,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstarta,indenda,icset,icset,icset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid )
            {
                logtma /= 3.0;
                logtmb /= 3.0;
                logtmc /= 3.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                tma *= tmc;

                T sf; sf = 1.0/3.0;

                safedivby(res,pow(tma,sf)); //res /= pow(tma,sf);
            }
        }

        else if ( ( xainfo.xusize() == 2 ) && ( xcinfo.xusize() == 2 ) )
        {
            T tma;
            T tmb;
            T tmc;
            T tmd;

            T logtma;
            T logtmb;
            T logtmc;
            T logtmd;

            int logtmavalid = 0;
            int logtmbvalid = 0;
            int logtmcvalid = 0;
            int logtmdvalid = 0;

            KK2(     res,logres,logresvalid,xa,xc,xainfo,xcinfo,bias,NULL,ia,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,icset);
            KK2(atempres,logres,logresvalid,xb,xd,xbinfo,xdinfo,bias,NULL,ib,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstartb,indendb,ibset,idset);

            res *= atempres;

            KK2(tma,logtma,logtmavalid,xa,xa,xainfo,xainfo,bias,NULL,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,iaset);
            KK2(tmc,logtmc,logtmcvalid,xc,xc,xcinfo,xcinfo,bias,NULL,ic,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,ibset,ibset);
            KK2(tmb,logtmb,logtmbvalid,xb,xb,xbinfo,xbinfo,bias,NULL,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstartb,indendb,icset,icset);
            KK2(tmd,logtmd,logtmdvalid,xd,xd,xdinfo,xdinfo,bias,NULL,id,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstartb,indendb,idset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid && logtmcvalid && logtmdvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;
                logtmc /= 2.0;
                logtmd /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;
                res -= logtmc;
                res -= logtmd;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                tma *= tmc;
                tma *= tmd;

                T sf; sf = 1.0/2.0;

                safedivby(res,pow(tma,sf)); //res /= pow(tma,sf);
            }
        }

        else if ( xainfo.xusize() == 3 )
        {
            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid = 0;
            int logtmbvalid = 0;

            KK2(res,logres,logresvalid,xa,xd,xainfo,xdinfo,bias,NULL,ia,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,idset);

            KK2(tma,logtma,logtmavalid,xa,xa,xainfo,xainfo,bias,NULL,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,iaset);
            KK2(tmb,logtmb,logtmbvalid,xd,xd,xdinfo,xdinfo,bias,NULL,id,id,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,idset,idset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else if ( xbinfo.xusize() == 3 )
        {
            T tma;
            T tmb;

            T logtma;
            T logtmb;

            int logtmavalid = 0;
            int logtmbvalid = 0;

            KK2(res,logres,logresvalid,xa,xb,xainfo,xbinfo,bias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,ibset);

            KK2(tma,logtma,logtmavalid,xa,xa,xainfo,xainfo,bias,NULL,ia,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,iaset,iaset);
            KK2(tmb,logtmb,logtmbvalid,xb,xb,xbinfo,xbinfo,bias,NULL,ib,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstarta,indenda,ibset,ibset);

            if ( logresvalid && logtmavalid && logtmbvalid )
            {
                logtma /= 2.0;
                logtmb /= 2.0;

                res = logres;

                res -= logtma;
                res -= logtmb;

                OP_exp(res);
            }

            else
            {
                tma *= tmb;
                OP_sqrt(tma);
                safedivby(res,tma); //res /= tma;
            }
        }

        else
        {
            res = 1.0;
        }
    }

    else
    {
        KK4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,indstart,indend,iaset,ibset,icset,idset);
    }

    return res;
}

//phantomx
template <class T>
T &MercerKernel::xKKm(int m, T &res,
                     Vector<const SparseVector<gentype> *> &x,
                     Vector<const vecInfo *> &xinfo,
                     const T &bias,
                     Vector<int> &i,
                     const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid,
                     const Matrix<double> *xy, int justcalcip, const Vector<int> *xiset, int indstart, int indend, int ns) const
{
    T logres; logres = 0.0;
    T tempres; tempres = 0.0;
    int logresvalid = 0;
    //int ns = numSplits();
    int q,s,r = 0;

    if ( isfullnorm && !ns && !justcalcip )
    {
        int effm = 0;
        int ii,jj,kk=0;

        for ( ii = 0 ; ii < m ; ii += (*(xinfo(ii))).xusize() )
        {
            effm++;
        }

        NiceAssert( ii == m );

        T tma; tma = 1.0;
        T tmb;

        KKm(m,res,logres,logresvalid,x,xinfo,bias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,xy,NULL,justcalcip,indstart,indend,xiset);

        T logtma; logtma = 0.0; // must be zero
        T logtmb; logtmb = 0.0;

        int logtmavalid = 1; // deliberately 1
        int logtmbvalid = 0;

        for ( ii = 0 ; ii < m ; ii += (*(xinfo(ii))).xusize() )
        {
            int usize = (*(xinfo(ii))).xusize();
            int mcsize = usize*effm;

            Vector<const SparseVector<gentype> *> xa(mcsize);
            Vector<const vecInfo *> xainfo(mcsize);
            Vector<int> ia(mcsize);
            Vector<int> iaset(mcsize);

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;
            retVector<int>                           tmpvc;
            retVector<int>                           tmpvd;

            for ( jj = 0 ; jj < usize ; jj++ )
            {
                xa("&",jj*effm,1,((jj+1)*effm)-1,tmpva)     = x(ii+jj);
                xainfo("&",jj*effm,1,((jj+1)*effm)-1,tmpvb) = xinfo(ii+jj);
                ia("&",jj*effm,1,((jj+1)*effm)-1,tmpvc)     = i(ii+jj);
                iaset("&",jj*effm,1,((jj+1)*effm)-1,tmpvd)  = (*xiset)(ii+jj);
            }

            logtmbvalid = 0;

            KKm(mcsize,tmb,logtmb,logtmbvalid,xa,xainfo,bias,ia,pxyprod,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstart,indend,&iaset);

            tma *= tmb;
            logtma += logtmb;
            logtmavalid *= logtmbvalid;

            kk++;
        }

        if ( logresvalid && logtmavalid )
        {
            logtma /= effm;

            res  = logres;
            res -= logtma;

            OP_exp(res);
        }

        else
        {
            T oneonm; oneonm = 1.0/effm;

            tma = pow(tma,oneonm);
            safedivby(res,tma); //res /= tma;
        }
    }

    else if ( !isfullnorm && ns && !justcalcip )
    {
        Vector<int> splitPoint(ns);

        if ( ns )
        {
            for ( q = indstart ; q < indend ; q++ )
            {
                if ( isSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        int maxusize = 1;

        for ( q = 0 ; q < m ; q++ )
        {
            if ( (*(xinfo(q))).xusize() > maxusize )
            {
                maxusize = (*(xinfo(q))).xusize();
            }
        }

        Vector<Vector<const SparseVector<gentype> *> > xx(maxusize);
        Vector<Vector<const vecInfo *> > xxinfo(maxusize);
        Vector<Vector<int> > ii(maxusize);
        Vector<Vector<int> > iiset(maxusize);

        for ( q = 0 ; q < m ; )
        {
            int usize = (*(xinfo(q))).xusize();

            for ( r = 0 ; r < usize ; q++,r++ )
            {
                s = xx(r).size();

                xx("&",r).add(s);     xx("&",r)("&",s)     = x(q);
                xxinfo("&",r).add(s); xxinfo("&",r)("&",s) = xinfo(q);
                ii("&",r).add(s);     ii("&",r)("&",s)     = i(q);
                iiset("&",r).add(s);  iiset("&",r)("&",s)  = xiset ? (*xiset)(q) : 0;
            }
        }

        for ( r = 0 ; r < maxusize ; r++ )
        {
            int indstartq = indstart;
            int indendq   = indend;

            if ( r%(ns+1) == 0 )
            {
                indstartq = indstart;
                indendq   = splitPoint(r%(ns+1));
            }

            else if ( r%(ns+1) < ns )
            {
                indstartq = splitPoint((r-1)%(ns+1))+1;
                indendq   = splitPoint(r%(ns+1));
            }

            else
            {
                indstartq = splitPoint((r-1)%(ns+1))+1;
                indendq   = indend;
            }

            if ( r == 0 )
            {
                KKm(xx(r).size(),res,logres,logresvalid,xx("&",r),xxinfo("&",r),bias,ii("&",r),NULL,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstartq,indendq,&(iiset("&",r)));
            }

            else
            {
                KKm(xx(r).size(),tempres,logres,logresvalid,xx("&",r),xxinfo("&",r),bias,ii("&",r),NULL,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstartq,indendq,&(iiset("&",r)));

                res *= tempres;
            }
        }
    }

    else if ( isfullnorm && ns && !justcalcip )
    {
        Vector<int> splitPoint(ns);

        if ( ns )
        {
            for ( q = indstart ; q < indend ; q++ )
            {
                if ( isSplit(q) )
                {
                    splitPoint("&",r) = q;
                    r++;
                }
            }
        }

        int maxusize = 1;

        for ( q = 0 ; q < m ; q++ )
        {
            if ( (*(xinfo(q))).xusize() > maxusize )
            {
                maxusize = (*(xinfo(q))).xusize();
            }
        }

        Vector<Vector<const SparseVector<gentype> *> > xx(maxusize);
        Vector<Vector<const vecInfo *> > xxinfo(maxusize);
        Vector<Vector<int> > ii(maxusize);
        Vector<Vector<int> > iiset(maxusize);

        for ( q = 0 ; q < m ; )
        {
            int usize = (*(xinfo(q))).xusize();

            for ( r = 0 ; r < usize ; q++,r++ )
            {
                s = xx(r).size();

                xx("&",r).add(s);     xx("&",r)("&",s)     = x(q);
                xxinfo("&",r).add(s); xxinfo("&",r)("&",s) = xinfo(q);
                ii("&",r).add(s);     ii("&",r)("&",s)     = i(q);
                iiset("&",r).add(s);  iiset("&",r)("&",s)  = xiset ? (*xiset)(q) : 0;
            }
        }

        res = 1.0;

        for ( r = 0 ; r < maxusize ; r++ )
        {
            int indstartq = indstart;
            int indendq   = indend;

            if ( r%(ns+1) == 0 )
            {
                indstartq = indstart;
                indendq   = splitPoint(r%(ns+1));
            }

            else if ( r%(ns+1) < ns )
            {
                indstartq = splitPoint((r-1)%(ns+1))+1;
                indendq   = splitPoint(r%(ns+1));
            }

            else
            {
                indstartq = splitPoint((r-1)%(ns+1))+1;
                indendq   = indend;
            }

            int locm = xx(r).size();

            if ( locm > 1 )
            {
                KKm(locm,tempres,logres,logresvalid,xx("&",r),xxinfo("&",r),bias,ii("&",r),NULL,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstartq,indendq,&(iiset("&",r)));

                T tma;
                T logtma;
                int logtmavalid = 0;
                Vector<int> repind(locm);
                int jj;

                retVector<const SparseVector<gentype> *> tmpva;
                retVector<const vecInfo *>               tmpvb;
                retVector<int>                           tmpvc;

                for ( jj = 0 ; jj < locm ; jj++ )
                {
                    repind = jj;

                    KKm(locm,tma,logtma,logtmavalid,xx("&",r)("&",repind,tmpva),xxinfo("&",r)("&",repind,tmpvb),bias,ii("&",r)("&",repind,tmpvc),NULL,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstartq,indendq,&(iiset("&",r)));

                    if ( logresvalid && logtmavalid )
                    {
                        logtma /= ((double) locm);

                        tempres  = logres;
                        tempres -= logtma;

                        OP_exp(tempres);
                    }

                    else
                    {
                        logresvalid = 0;

                        T sf; sf = 1.0/((double) locm);
                        safedivby(tempres,pow(tma,sf)); //tempres /= pow(tma,sf);
                    }
                }

                res *= tempres;
            }
        }
    }

    else
    {
        KKm(m,res,logres,logresvalid,x,xinfo,bias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,xy,NULL,justcalcip,indstart,indend,xiset);
    }

    return res;
}






















inline int isgentype(const gentype &ind);
inline int isgentype(const double  &ind);

inline int isgentype(const gentype &ind)
{
    (void) ind;

    return 1;
}

inline int isgentype(const double &ind)
{
    (void) ind;

    return 0;
}

inline int isiteqn(const gentype &ind);
inline int isiteqn(const double  &ind);

inline int isiteqn(const gentype &ind)
{
    return ind.isValEqn();
}

inline int isiteqn(const double &ind)
{
    (void) ind;

    return 0;
}

inline
int MercerKernel::subSample(SparseVector<SparseVector<gentype> > &subval, gentype &a) const
{
    int res = 0;

    Vector<gentype> locdist(xsampdist);

    NiceAssert( locdist.size() == xindsub.size() );

    int j;

    for ( j = 0 ; j < locdist.size() ; j++ )
    {
        locdist("&",j).finalise();
        subval("&",zeroint())("&",xindsub(j)) = locdist(j);
        subval("&",zeroint())("&",xindsub(j)).finalise();
    }

    res += a.substitute(subval);
    res += a.finalise();

    return res;
}

inline
int MercerKernel::subSample(SparseVector<SparseVector<gentype> > &subval, double &a) const
{
    int res = 0;

    (void) subval;
    (void) a;

    throw("This is weird.");

    return res;
}

inline
int MercerKernel::subSample(SparseVector<SparseVector<gentype> > &subval, SparseVector<gentype> &x, vecInfo &xinfo) const
{
    int res = 0;

    Vector<gentype> locdist(xsampdist);

    NiceAssert( locdist.size() == xindsub.size() );

    int j;

    for ( j = 0 ; j < locdist.size() ; j++ )
    {
        locdist("&",j).finalise();
        subval("&",zeroint())("&",xindsub(j)) = locdist(j);
        subval("&",zeroint())("&",xindsub(j)).finalise();
    }

    for ( j = 0 ; j < x.indsize() ; j++ )
    {
        res += x.direref(j).substitute(subval);
        res += x.direref(j).finalise();
    }

    getvecInfo(xinfo,x);

    return res;
}
























//phantomx
template <class T>
T &MercerKernel::KK0(T &res, T &logres, int &logresvalid,
                     const T &bias,
                     const gentype **pxyprod,
                     int xdim, int xconsist, int assumreal, int resmode, int mlid,
                     int justcalcip, int indstart, int indend, int skipbias) const
{
    if ( !skipbias && isiteqn(bias) )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int q;
        int maxq = numSamples();

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( q = 0 ; q < maxq ; q++ )
        {
            gentype gbias(bias);
            gentype gres,glogres;

            if ( !subSample(subval,gbias) && !q )
            {
                goto postbias;
            }

            KK0(gres,glogres,logresvalid,gbias,NULL,xdim,xconsist,assumreal,resmode,mlid,justcalcip,indstart,indend,1);

            if ( !q ) { res =  (T) gres; }
            else      { res += (T) gres; }
        }

        res /= maxq;

        logresvalid = 0;

        return res;
    }

postbias:

    LL0(res,logres,logresvalid,bias,pxyprod,xdim,xconsist,assumreal,resmode,mlid,justcalcip,indstart,indend);

    return res;
}

//phantomx
template <class T>
T &MercerKernel::KK1(T &res, T &logres, int &logresvalid,
                     const SparseVector<gentype> &xa, 
                     const vecInfo &xainfo, 
                     const T &bias,
                     const gentype **pxyprod,
                     int ia,  
                     int xdim, int xconsist, int assumreal, int resmode, int mlid,
                     const double *xy, int justcalcip, int indstart, int indend, 
                     int iaset, 
                     int skipbias, 
                     int skipxa) const
{
    if ( !skipbias && isiteqn(bias) )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qb;
        int maxq = numSamples();

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qb = 0 ; qb < maxq ; qb++ )
        {
            gentype gbias(bias);

            if ( !subSample(subval,gbias) && !qb )
            {
                goto postbias;
            }

            gentype gres,glogres;

            KK1(gres,glogres,logresvalid,xa,xainfo,gbias,NULL,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstart,indend,iaset,1,skipxa);

            if ( !qb ) { res =  (T) gres; }
            else       { res += (T) gres; }
        }

        res /= maxq;

        logresvalid = 0;

        return res;
    }

postbias:

    if ( !skipxa && xainfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxa(xa);
            vecInfo xxainfo;

            if ( !subSample(subval,xxa,xxainfo) && !qxa )
            {
                goto postxa;
            }

            gentype gres,glogres;

            KK1(gres,glogres,logresvalid,xxa,xxainfo,gbias,NULL,ia,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,indstart,indend,iaset,skipbias,1);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !iaset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !iaset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxa:

    LL1(res,logres,logresvalid,xa,xainfo,bias,pxyprod,ia,xdim,xconsist,assumreal,resmode,mlid,xy,justcalcip,indstart,indend);

    return res;
}

//phantomx
template <class T>
T &MercerKernel::KK2(T &res, T &logres, int &logresvalid,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                     const vecInfo &xainfo, const vecInfo &xbinfo,
                     const T &bias,
                     const gentype **pxyprod,
                     int ia, int ib,
                     int xdim, int xconsist, int assumreal, int resmode, int mlid,
                     const double *xy00, const double *xy10, const double *xy11, int justcalcip, int indstart, int indend, 
                     int iaset, int ibset,
                     int skipbias,
                     int skipxa, int skipxb) const
{
    if ( !skipbias && isiteqn(bias) )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qb;
        int maxq = numSamples();

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qb = 0 ; qb < maxq ; qb++ )
        {
            gentype gbias(bias);

            if ( !subSample(subval,gbias) && !qb )
            {
                goto postbias;
            }

            gentype gres,glogres;

            KK2(gres,glogres,logresvalid,xa,xb,xainfo,xbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,ibset,1,skipxa,skipxb);

            if ( !qb ) { res =  (T) gres; }
            else       { res += (T) gres; }
        }

        res /= maxq;

        logresvalid = 0;

        return res;
    }

postbias:

    if ( !skipxa && xainfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxa(xa);
            vecInfo xxainfo;

            if ( !subSample(subval,xxa,xxainfo) && !qxa )
            {
                goto postxa;
            }

            gentype gres,glogres;

            KK2(gres,glogres,logresvalid,xxa,xb,xxainfo,xbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,ibset,skipbias,1,skipxb);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !iaset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !iaset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxa:

    if ( !skipxb && xbinfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxb(xb);
            vecInfo xxbinfo;

            if ( !subSample(subval,xxb,xxbinfo) && !qxa )
            {
                goto postxb;
            }

            gentype gres,glogres;

            KK2(gres,glogres,logresvalid,xa,xxb,xainfo,xxbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,ibset,skipbias,skipxa,1);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !ibset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !ibset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxb:

    LL2(res,logres,logresvalid,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,justcalcip,indstart,indend);

    return res;
}

//phantomx
template <class T>
T &MercerKernel::KK3(T &res, T &logres, int &logresvalid,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                     const T &bias,
                     const gentype **pxyprod,
                     int ia, int ib, int ic, 
                     int xdim, int xconsist, int assumreal, int resmode, int mlid,
                     const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const Vector<int> *s, int justcalcip, int indstart, int indend, 
                     int iaset, int ibset, int icset,
                     int skipbias,
                     int skipxa, int skipxb, int skipxc) const
{
    if ( !skipbias && isiteqn(bias) )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qb;
        int maxq = numSamples();

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qb = 0 ; qb < maxq ; qb++ )
        {
            gentype gbias(bias);

            if ( !subSample(subval,gbias) && !qb )
            {
                goto postbias;
            }

            gentype gres,glogres;

            KK3(gres,glogres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,gbias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,ibset,icset,1,skipxa,skipxb,skipxc);

            if ( !qb ) { res =  (T) gres; }
            else       { res += (T) gres; }
        }

        res /= maxq;

        logresvalid = 0;

        return res;
    }

postbias:

    if ( !skipxa && xainfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxa(xa);
            vecInfo xxainfo;

            if ( !subSample(subval,xxa,xxainfo) && !qxa )
            {
                goto postxa;
            }

            gentype gres,glogres;

            KK3(gres,glogres,logresvalid,xxa,xb,xc,xxainfo,xbinfo,xcinfo,gbias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,ibset,icset,skipbias,1,skipxb,skipxc);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !iaset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !iaset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxa:

    if ( !skipxb && xbinfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxb(xb);
            vecInfo xxbinfo;

            if ( !subSample(subval,xxb,xxbinfo) && !qxa )
            {
                goto postxb;
            }

            gentype gres,glogres;

            KK3(gres,glogres,logresvalid,xa,xxb,xc,xainfo,xxbinfo,xcinfo,gbias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,ibset,icset,skipbias,skipxa,1,skipxc);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !ibset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !ibset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxb:

    if ( !skipxc && xcinfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxc(xc);
            vecInfo xxcinfo;

            if ( !subSample(subval,xxc,xxcinfo) && !qxa )
            {
                goto postxc;
            }

            gentype gres,glogres;

            KK3(gres,glogres,logresvalid,xa,xb,xxc,xainfo,xbinfo,xxcinfo,gbias,NULL,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,NULL,NULL,NULL,NULL,NULL,justcalcip,indstart,indend,iaset,ibset,icset,skipbias,skipxa,skipxb,1);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !icset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !icset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxc:

    LL3(res,logres,logresvalid,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,s,justcalcip,indstart,indend);

    return res;
}

//phantomx
template <class T>
T &MercerKernel::KK4(T &res, T &logres, int &logresvalid,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                     const T &bias,
                     const gentype **pxyprod,
                     int ia, int ib, int ic, int id, 
                     int xdim, int xconsist, int assumreal, int resmode, int mlid,
                     const Vector<int> *s, int justcalcip,
                     const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int indstart, int indend, 
                     int iaset, int ibset, int icset, int idset,
                     int skipbias,
                     int skipxa, int skipxb, int skipxc, int skipxd) const
{
    if ( !skipbias && isiteqn(bias) )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qb;
        int maxq = numSamples();

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qb = 0 ; qb < maxq ; qb++ )
        {
            gentype gbias(bias);

            if ( !subSample(subval,gbias) && !qb )
            {
                goto postbias;
            }

            gentype gres,glogres;

            KK4(gres,glogres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,gbias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend,iaset,ibset,icset,idset,1,skipxa,skipxb,skipxc,skipxd);

            if ( !qb ) { res =  (T) gres; }
            else       { res += (T) gres; }
        }

        res /= maxq;

        logresvalid = 0;

        return res;
    }

postbias:

    if ( !skipxa && xainfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxa(xa);
            vecInfo xxainfo;

            if ( !subSample(subval,xxa,xxainfo) && !qxa )
            {
                goto postxa;
            }

            gentype gres,glogres;

            KK4(gres,glogres,logresvalid,xxa,xb,xc,xd,xxainfo,xbinfo,xcinfo,xdinfo,gbias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend,iaset,ibset,icset,idset,skipbias,1,skipxb,skipxc,skipxd);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !iaset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !iaset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxa:

    if ( !skipxb && xbinfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxb(xb);
            vecInfo xxbinfo;

            if ( !subSample(subval,xxb,xxbinfo) && !qxa )
            {
                goto postxb;
            }

            gentype gres,glogres;

            KK4(gres,glogres,logresvalid,xa,xxb,xc,xd,xainfo,xxbinfo,xcinfo,xdinfo,gbias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend,iaset,ibset,icset,idset,skipbias,skipxa,1,skipxc,skipxd);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !ibset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !ibset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxb:

    if ( !skipxc && xcinfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxc(xc);
            vecInfo xxcinfo;

            if ( !subSample(subval,xxc,xxcinfo) && !qxa )
            {
                goto postxc;
            }

            gentype gres,glogres;

            KK4(gres,glogres,logresvalid,xa,xb,xxc,xd,xainfo,xbinfo,xxcinfo,xdinfo,gbias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend,iaset,ibset,icset,idset,skipbias,skipxa,skipxb,1,skipxd);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !icset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !icset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxc:

    if ( !skipxd && xdinfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxd(xd);
            vecInfo xxdinfo;

            if ( !subSample(subval,xxd,xxdinfo) && !qxa )
            {
                goto postxd;
            }

            gentype gres,glogres;

            KK4(gres,glogres,logresvalid,xa,xb,xc,xxd,xainfo,xbinfo,xcinfo,xxdinfo,gbias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,NULL,justcalcip,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend,iaset,ibset,icset,idset,skipbias,skipxa,skipxb,skipxc,1);

            if ( !qxa )        { res =  (T) gres; }
            else if ( !idset ) { res += (T) gres; }
            else               { res =  ( ( (T) gres ) > ( (T) res ) ) ? ( (T) gres ) : ( (T) res ); }
        }

        if ( !idset )
        {
            res /= maxq;
        }

        logresvalid = 0;

        return res;
    }

postxd:

    LL4(res,logres,logresvalid,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,s,justcalcip,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,indstart,indend);

    return res;
}

template <class T>
T &MercerKernel::KK6(T &res, T &logres, int &logresvalid,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const SparseVector<gentype> &xe, const SparseVector<gentype> &xf, 
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const vecInfo &xeinfo, const vecInfo &xfinfo, 
                     const T &bias, 
                     const gentype **pxyprod, 
                     int ia, int ib, int ic, int id, int ie, int jf, 
                     int xdim, int xconsist, int assumreal, int xresmode, int mlid,
                     const Matrix<double> *xy, const Vector<int> *s, int justcalcip, int indstart, int indend, 
                     int iaset, int ibset, int icset, int idset, int ieset, int ifset) const
{
    (void) pxyprod;
    (void) s;
    (void) xy;
    (void) justcalcip;

    Vector<const SparseVector<gentype> *> x(6);
    Vector<const vecInfo *> xinfo(6);
    Vector<int> i(6);
    Vector<int> iset(6);

    int z = 0;

    x("&",z)     = &xa;
    xinfo("&",z) = &xainfo;
    i("&",z)     = ia;
    iset("&",z)  = iaset;

    x("&",1)     = &xb;
    xinfo("&",1) = &xbinfo;
    i("&",1)     = ib;
    iset("&",1)  = ibset;

    x("&",2)     = &xc;
    xinfo("&",2) = &xcinfo;
    i("&",2)     = ic;
    iset("&",2)  = icset;

    x("&",3)     = &xd;
    xinfo("&",3) = &xdinfo;
    i("&",3)     = id;
    iset("&",3)  = idset;

    x("&",4)     = &xe;
    xinfo("&",4) = &xeinfo;
    i("&",4)     = ie;
    iset("&",4)  = ieset;

    x("&",5)     = &xf;
    xinfo("&",5) = &xfinfo;
    i("&",5)     = jf;
    iset("&",5)  = ifset;

    return KKm(6,res,logres,logresvalid,x,xinfo,bias,i,NULL,xdim,xconsist,assumreal,xresmode,mlid,NULL,NULL,justcalcip,indstart,indend,&iset);
}

//phantomx
template <class T>
T &MercerKernel::KKm(int m, T &res, T &logres, int &logresvalid,
                     Vector<const SparseVector<gentype> *> &x,
                     Vector<const vecInfo *> &xinfo,
                     const T &bias,
                     Vector<int> &i,
                     const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid,
                     const Matrix<double> *xy, const Vector<int> *s, int justcalcip, int indstart, int indend, 
                     const Vector<int> *iset,
                     int skipbias,
                     int skipx) const
{
    if ( !skipbias && isiteqn(bias) )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qb;
        int maxq = numSamples();

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qb = 0 ; qb < maxq ; qb++ )
        {
            gentype gbias(bias);

            if ( !subSample(subval,gbias) && !qb )
            {
                goto postbias;
            }

            gentype gres,glogres;

            KKm(m,gres,glogres,logresvalid,x,xinfo,gbias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstart,indend,iset,1,skipx);

            if ( !qb ) { res =  (T) gres; }
            else       { res += (T) gres; }
        }

        res /= maxq;

        logresvalid = 0;

        return res;
    }

postbias:

    Vector<int> isValEqn;

    if ( !skipx )
    {
        int qr;

        for ( qr = 0 ; qr < m ; qr++ )
        {
            if ( (*(xinfo(qr))).xiseqn() )
            {
                isValEqn.add(isValEqn.size());
                isValEqn("&",isValEqn.size()-1) = qr;
            }
        }
    }

    if ( !skipx && isValEqn.size() )
    {
        int q;

        Vector<SparseVector<gentype> *> yy(x.size());
        Vector<vecInfo *> yyinfo(xinfo.size());

        Vector<const SparseVector<gentype> *> xx(x);
        Vector<const vecInfo *> xxinfo(xinfo);

        SparseVector<SparseVector<gentype> > subval;

        int haschanged = 0;

        for ( q = 0 ; q < isValEqn.size() ; q++ )
        {
            MEMNEW(yy("&",isValEqn(q)),SparseVector<gentype>(*(x(isValEqn(q)))));
            MEMNEW(yyinfo("&",isValEqn(q)),vecInfo(*(xinfo(isValEqn(q)))));

            haschanged += subSample(subval,*(yy("&",isValEqn(q))),*(yyinfo("&",isValEqn(q))));

            xx("&",isValEqn(q))     = yy("&",isValEqn(q));
            xxinfo("&",isValEqn(q)) = yyinfo("&",isValEqn(q));

            //FIXME: need to implement setwise m-kernels
            NiceAssert( !iset || !(*iset)(q) );
        }

        if ( !haschanged )
        {
            for ( q = 0 ; q < isValEqn.size() ; q++ )
            {
                MEMDEL(yy("&",isValEqn(q)));
                MEMDEL(yyinfo("&",isValEqn(q)));
            }

            goto postx;
        }

        int isdone  = 0;
        int isfirst = 1;

        int maxq = numSamples();

        gentype gbias(bias);

        Vector<int> qx(isValEqn.size());

        qx = zeroint();

        while ( !isdone )
        {
            gentype gres,glogres;

            KKm(m,gres,glogres,logresvalid,xx,xxinfo,gbias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,justcalcip,indstart,indend,iset,skipbias,1);

            if ( isfirst ) { res =  (T) gres; }
            else           { res += (T) gres; }

            for ( q = 0 ; q < qx.size() ; q++ )
            {
                *(yy("&",isValEqn(q)))     = *(x(isValEqn(q)));
                *(yyinfo("&",isValEqn(q))) = *(xinfo(isValEqn(q)));

                subSample(subval,*(yy("&",isValEqn(q))),*(yyinfo("&",isValEqn(q))));

                xx("&",isValEqn(q))     = yy("&",isValEqn(q));
                xxinfo("&",isValEqn(q)) = yyinfo("&",isValEqn(q));

                qx("&",q)++;

                if ( qx(q) < maxq )
                {
                    break;
                }

                else
                {
                    qx("&",q) = 0;
                }
            }

            isdone  = ( q < qx.size() ) ? 0 : 1;
            isfirst = 0;
        }

        res /= pow(maxq,isValEqn.size());

        for ( q = 0 ; q < isValEqn.size() ; q++ )
        {
            MEMDEL(yy("&",isValEqn(q)));
            MEMDEL(yyinfo("&",isValEqn(q)));
        }

        return res;
    }

postx:

    LLm(m,res,logres,logresvalid,x,xinfo,bias,i,pxyprod,xdim,xconsist,assumreal,resmode,mlid,xy,s,justcalcip,indstart,indend);

    return res;
}






























//phantomx
template <class T>
T &MercerKernel::LL0(T &res, T &logres, int &logresvalid,
                     const T &bias,
                     const gentype **pxyprod,
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, int justcalcip, int indstart, int indend) const
{
    NiceAssert( ! ( isFastKernelSum() && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) );

    logresvalid = 0;

    if ( !size() )
    {
        return res = 0.0;
    }

    if ( ( ( isAltDiff() >= 100 ) && ( isAltDiff() <= 199 ) ) && !justcalcip )
    {
        res = 0;
    }

    else if ( ( resmode & 0x01 ) )
    {
        NiceAssert( !isfullnorm );
        NiceAssert( !justcalcip );
        NiceAssert( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) );
        NiceAssert( !( isprod ) );

        gentype tempres;
        gentype dummyval(0);

        K0i(tempres,dummyval,xdim,0,resmode,mlid,indstart,indend);

        res = (T) tempres;
    }

    else if ( isprod )
    {
        NiceAssert( !( resmode & 0x80 ) );

        // Should probably replace this with a proper fast version

        Vector<const SparseVector<gentype> *> xx(0);
        Vector<const vecInfo *> xxinfo(0);
        Vector<int> ii(0);

        LLm(0,res,logres,logresvalid,xx,xxinfo,bias,ii,pxyprod,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,0,indstart,indend);
    }

    else if ( ( isFastKernelSum() || isFastKernelChain() ) && ( resmode && 0x80 ) && !justcalcip )
    {
        res = 0;
    }

    else if ( isFastKernelSum() || isFastKernelChain() || ( isFastKernelXfer() && !resmode ) || justcalcip )
    {
        int locindstart = (isFastKernelXfer()?1:indstart); // NB: isFastKernelXfer() implies indstart == 0
        int locindend   = indend; //size()-1;

        T prod;   prod   = 0.0;
        T diffis; diffis = 0.0;

        // Calculate inner products and differences if required

        if ( justcalcip )
        {
            prod = 0.0;
        }

        else
        {
            prod = bias;
        }

        if ( !( isFastKernelSum() || isFastKernelChain() || justcalcip ) )
        {
            int dummyind = 0;

            kernel8xx(0,prod,dummyind,cType(zeroint()),xdim,0,0,mlid);

            if ( !justcalcip )
            {
                prod *= (const T &) cWeight(0);
            }
        }

        if ( justcalcip )
        {
            res = prod;
        }

        else
        {
            if ( isNormalised(locindend) )
            {
                res = 1.0;
            }

            else
            {
                T dummy;

                logresvalid = KKpro(res,prod,diffis,NULL,locindstart,locindend,xdim,0,logres,&dummy);
            }
        }
    }

    else
    {
        NiceAssert( ismagterm == zeroint() );
        NiceAssert( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) );

        gentype xyprod(0.0);

        if ( pxyprod && pxyprod[0] )
        {
            xyprod = *pxyprod[0];
        }

        else
        {
            xyprod += bias;
        }

        gentype tempres;

        K0i(tempres,xyprod,xdim,0,resmode,mlid,indstart,indend);

        res = (T) tempres;
    }

    return res;
}

//phantomx
template <class T>
T &MercerKernel::LL1(T &res, T &logres, int &logresvalid,
                     const SparseVector<gentype> &xa, 
                     const vecInfo &xainfo, 
                     const T &bias,
                     const gentype **pxyprod,
                     int ia,  
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy, int justcalcip, int indstart, int indend) const
{
    NiceAssert( ! ( isFastKernelSum() && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) );

    (void) xy;

    logresvalid = 0;

    if ( !size() )
    {
        return res = 0.0;
    }

    if ( ( isAltDiff() == 300 ) && !justcalcip )
    {
        goto badout;
    }

    else if ( ( isAltDiff() >= 100 ) && ( isAltDiff() <= 199 ) && !justcalcip )
    {
        goto badout;
    }

    else if ( ( resmode & 0x01 ) )
    {
        goto badout;
    }

    else if ( isprod )
    {
        NiceAssert( !justcalcip );
        NiceAssert( !( resmode & 0x80 ) );

        // Should probably replace this with a proper fast version

        Vector<const SparseVector<gentype> *> xx(1);
        Vector<const vecInfo *> xxinfo(1);
        Vector<int> ii(1);

        xx("[]",zeroint()) = &xa;
        xxinfo("[]",zeroint()) = &xainfo;
        ii("[]",zeroint()) = ia;

        LLm(1,res,logres,logresvalid,xx,xxinfo,bias,ii,pxyprod,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,0,indstart,indend);
    }

    else if ( ( isFastKernelSum() || isFastKernelChain() ) && ( resmode && 0x80 ) && !justcalcip )
    {
        res = 0;
    }

    else if ( isFastKernelSum() || isFastKernelChain() || ( isFastKernelXfer() && !resmode ) || justcalcip )
    {
        int locindstart = (isFastKernelXfer()?1:indstart); // NB: isFastKernelXfer() implies indstart == 0
        int locindend   = indend; //size()-1;

        int needxxprod = isNormalised(locindend);

        T xyprod; xyprod = 0.0;
        T diffis; diffis = 0.0;
        T xaprod; xaprod = 0.0;

        // Calculate inner products and differences if required

        if ( isFastKernelSum() || isFastKernelChain() || justcalcip )
        {
            if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
            }

            else if ( needsInner(0,1) || ( isFastKernelSum() && needsInner(-1,1) ) || justcalcip )
            {
                oneProductDiverted(xyprod,xa,xconsist,assumreal);

                if ( !justcalcip )
                {
                    xyprod += bias;
                }
            }

            if ( needxxprod && !justcalcip )
            {
                oneProductDiverted(xaprod,xa,xconsist,assumreal);

                xaprod += bias;
            }

            if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else if ( ( ( needsDiff(0) || ( isFastKernelSum() && needsDiff() ) ) && ( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) ) ) && !justcalcip )
            {
                // Calculate ||x-y||^2 only as required

                diff1norm(diffis,xyprod,getmnorm(xainfo,xa,1,xconsist,assumreal));
            }

            else if ( ( needsDiff(0) && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) && !justcalcip )
            {
                goto badout;
            }
        }

        else
        {
            // ( isFastKernelXfer() && !resmode )
            // isSimpleFastKernelChain

            if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
            }

            else
            {
                int dummyind = 0;

                kernel8xx(0,xyprod,dummyind,cType(zeroint()),xa,xainfo,ia,xdim,0,0,mlid);
            }

            if ( needxxprod && !justcalcip )
            {
                int dummyind = 0;

                kernel8xx(0,xaprod,dummyind,cType(zeroint()),xa,xainfo,ia,xdim,0,0,mlid);
            }

            if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else if ( ( size() >= 2 ) && needsDiff(1) && ( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) ) && !justcalcip )
            {
                int dummyind = 0;

                T xanorm;

                if ( isAltDiff() == 0 )
                {
                    kernel8xx(0,xanorm,dummyind,cType(zeroint()),xa,xainfo,ia,xdim,0,0,mlid);
                }

                else
                {
                    kernel8xx(0,xanorm,dummyind,cType(zeroint()),xa,xa,xainfo,xainfo,ia,ia,xdim,0,0,mlid);
                }

                diff1norm(diffis,xyprod,xanorm);

                if ( !justcalcip )
                {
                    xyprod *= (const T &) cWeight(0);
                    xaprod *= (const T &) cWeight(0);
                    diffis *= (const T &) cWeight(0);
                }
            }

            else if ( needsDiff(0) && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) && !justcalcip )
            {
                goto badout;
            }
        }

        if ( justcalcip )
        {
            res = xyprod;
        }

        else
        {
            if ( isNormalised(locindend) )
            {
                T dummy;

                KKpro(res,xyprod,diffis,&ia,locindstart,locindend,xdim,1,dummy,&dummy);

                T xares;
                T zerodiff; zerodiff = 0.0;

                KKpro(xares,xaprod,zerodiff,&ia,locindstart,locindend,xdim,1,dummy,&dummy);

                safedivby(res,xares); //res /= xares;
            }

            else
            {
                T dummy;

                logresvalid = KKpro(res,xyprod,diffis,&ia,locindstart,locindend,xdim,1,logres,&dummy);
            }
        }
    }

    else
    {
        goto badout;
    }

    return res;

badout:
    // Design decision: in ml_base.cc, if d = 0 for one of the vectors
    // referenced here then this element will never be used.  Moreover there
    // are cases (eg isAltDiff set >1 with back-referenced data) where the
    // element is not properly defined but will never be used, so what we 
    // need to do is set it 0.  However having a "d = 0" catch will fail when
    // d starts non-zero, is set zero, then set non-zero, as will happen for
    // example when calculating LOO, n-fold error etc.  In such cases you need
    // to call a reset on that row/column, but you can't do that because (a) the
    // reset often calls setd (hence infinite recursion) or (b) there is an 
    // implicit assumption that Gp is independent of d (eg in semicopy functions
    // that retain the caches for speed in LOO, n-fold calculation).  Hence I've 
    // made the decision to return 0 here to avoid a whole stack of potential
    // coding complications at the price of possible silent failure if you set
    // somthing incorrectly.

    errstream() << "!!!1!!!";

    res = 0.0;

    return res;
}

inline int isCastableToRealWithoutLoss(const double &x)
{
    (void) x;

    return 1;
}

inline int isCastableToRealWithoutLoss(const gentype &x)
{
    return x.isCastableToRealWithoutLoss();
}

template <class T>
T &MercerKernel::LL2(T &res, T &logres, int &logresvalid,
                     const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                     const vecInfo &xinfo, const vecInfo &yinfo,
                     const T &bias,
                     const gentype **pxyprod,
                     int i, int j,
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, const double *xy00, const double *xy10, const double *xy11, int justcalcip, int indstart, int indend) const
{
    NiceAssert( ! ( isFastKernelSum() && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) );
    NiceAssert( ( !xy00 && !xy10 && !xy11 ) || !justcalcip );

    logresvalid = 0;

    if ( !size() )
    {
        return res = 0.0;
    }

    if ( ( resmode & 0x01 ) )
    {
        // Return equation form

        NiceAssert( !justcalcip );
        NiceAssert( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) );
        NiceAssert( !( isprod && !arexysimple(x,y) ) );

        gentype dummyres;

        K2i(dummyres,defaultgentype(),defaultgentype(),xinfo,yinfo,getmnorm(xinfo,x,2,xconsist,assumreal),getmnorm(yinfo,y,2,xconsist,assumreal),x,y,i,j,xdim,0,resmode,mlid,indstart,indend,assumreal);

        res = (T) dummyres;
    }

    else if ( isprod && !arexysimple(x,y) )
    {
        NiceAssert( !justcalcip );
        NiceAssert( !( resmode & 0x80 ) );

        // Kernel is of the form prod_i K(x_i,y_i)
        // x and y are not simple

        if ( xconsist && ( size() == 1 ) && isSimpleKernel() )
        {
            NiceAssert( x.nearindsize() == y.nearindsize() );

            SparseVector<gentype> xx;
            SparseVector<gentype> yy;

            vecInfo xxinfo;
            vecInfo yyinfo;

            int ii;

            T tempres;

            if ( x.nearindsize() )
            {
                for ( ii = 0 ; ii < x.nearindsize() ; ii++ )
                {
                    xx("&",zeroint()) = x.direcref(ii);
                    yy("&",zeroint()) = y.direcref(ii);

                    getvecInfo(xxinfo,xx,NULL,xconsist,assumreal);
                    getvecInfo(yyinfo,yy,NULL,xconsist,assumreal);

                    T dummya;
                    int dummyb;

                    LL2(tempres,dummya,dummyb,xx,yy,xxinfo,yyinfo,bias,NULL,i,j,1,1,assumreal,0,mlid,NULL,NULL,NULL,0,indstart,indend);

                    tempres /= (const T &) cWeight(0);

                    if ( !ii ) { res =  tempres; }
                    else       { res *= tempres; }
                }
            }

            else
            {
                res = 1;
            }
        }

        else
        {
            SparseVector<gentype> indres;

            combind(indres,x,y);

            SparseVector<gentype> xx;
            SparseVector<gentype> yy;

            vecInfo xxinfo;
            vecInfo yyinfo;

            int ii;

            T tempres;

            if ( indres.nearindsize() )
            {
                for ( ii = 0 ; ii < indres.nearindsize() ; ii++ )
                {
                    xx("&",zeroint()) = x(indres.ind(ii));
                    yy("&",zeroint()) = y(indres.ind(ii));

                    getvecInfo(xxinfo,xx,NULL,xconsist,assumreal);
                    getvecInfo(yyinfo,yy,NULL,xconsist,assumreal);

                    T dummya;
                    int dummyb;

                    LL2(tempres,dummya,dummyb,xx,yy,xxinfo,yyinfo,bias,NULL,i,j,1,1,assumreal,0,mlid,NULL,NULL,NULL,0,indstart,indend);

                    tempres /= (const T &) cWeight(0);

                    if ( !ii ) { res =  tempres; }
                    else       { res *= tempres; }
                }
            }

            else
            {
                res = 1;
            }
        }

        res *= (const T &) cWeight(0);
    }

    else if ( ( isFastKernelSum() || isFastKernelChain() ) && ( resmode && 0x80 ) && !justcalcip )
    {
        res = 0;
    }

    else if ( isFastKernelSum() || isFastKernelChain() || ( isFastKernelXfer() && !resmode ) || justcalcip )
    {
        int locindstart = (isFastKernelXfer()?1:indstart); // NB: isFastKernelXfer() implies indstart == 0
        int locindend   = indend; //size()-1;

        int needxxprod = isNormalised(locindend) || needsNorm(locindend);

        T xyprod; xyprod = 0.0;

        T yxprod; yxprod = 0.0;
        T xxprod; xxprod = 0.0;
        T yyprod; yyprod = 0.0;

        T diffis; diffis = 0.0;

        if ( isFastKernelSum() || isFastKernelChain() || justcalcip )
        {
            if ( xy10 && !justcalcip )
            {
                xyprod = *xy10;
                yxprod = *xy10;

                xyprod += bias;
                yxprod += bias;
            }

            else if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
                yxprod = xyprod;
            }

            else if ( needsInner(0,2) || ( isFastKernelSum() && needsInner(-1,2) ) || justcalcip )
            {
                twoProductDiverted(xyprod,x,y,xconsist,assumreal);
                twoProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

                if ( !justcalcip )
                {
                    xyprod += bias;
                    yxprod += bias;
                }
            }

            if ( needxxprod && !justcalcip )
            {
                if ( xy00 && xy11 )
                {
                    xxprod = (*xy00);
                    yyprod = (*xy11);

                    xxprod += bias;
                    yyprod += bias;
                }

                else if ( assumreal )
                {
                    xxprod = getmnorm(xinfo,x,2,xconsist,assumreal);
                    yyprod = getmnorm(yinfo,y,2,xconsist,assumreal);

                    xxprod += bias;
                    yyprod += bias;
                }

                else
                {
                    xxprod = getmnorm(xinfo,x,2,xconsist,assumreal);
                    yyprod = getmnorm(yinfo,y,2,xconsist,assumreal);

                    if ( !isCastableToRealWithoutLoss(xxprod) || !isCastableToRealWithoutLoss(yyprod) )
                    {
                        twoProductDiverted(xxprod,x,x,xconsist,assumreal);
                        twoProductDiverted(yyprod,y,y,xconsist,assumreal);
                    }

                    xxprod += bias;
                    yyprod += bias;
                }
            }

            if ( ( needsDiff(0) || ( isFastKernelSum() && needsDiff() ) ) && !justcalcip )
            {
                if ( pxyprod && pxyprod[1] )
                {
                    diffis = *pxyprod[1];
                }

                else
                {
                    // Calculate ||x-y||^2 only as required

                    if ( assumreal )
                    {
                        diff2norm(diffis,(double) xyprod,(double) getmnorm(xinfo,x,2,xconsist,assumreal),(double) getmnorm(yinfo,y,2,xconsist,assumreal));
                    }

                    else
                    {
                        diff2norm(diffis,(xyprod+yxprod)/2.0,getmnorm(xinfo,x,2,xconsist,assumreal),getmnorm(yinfo,y,2,xconsist,assumreal));
                    }
                }
            }

            xyprod += yxprod;
            xyprod *= 0.5;
        }

        else
        {
            // ( isFastKernelXfer() && !resmode )
            NiceAssert( !( resmode & 0x80 ) );

            if ( xy10 && !justcalcip )
            {
                xyprod = (*xy10);
            }

            else if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
            }

            else
            {
                int dummyind = 0;

                kernel8xx(0,xyprod,dummyind,cType(zeroint()),x,y,xinfo,yinfo,i,j,xdim,0,0,mlid);
            }

            if ( needxxprod && !justcalcip )
            {
                if ( xy00 && xy11 )
                {
                    xxprod = (*xy00);
                    yyprod = (*xy11);
                }

                else
                {
                    int dummyind = 0;

//errstream() << "phantomxyz mer 1\n";
                    kernel8xx(0,xxprod,dummyind,cType(zeroint()),x,x,xinfo,xinfo,i,i,xdim,0,0,mlid);
//errstream() << "phantomxyz mer 1b " << xxprod << "\n";
                    kernel8xx(0,yyprod,dummyind,cType(zeroint()),y,y,yinfo,yinfo,j,j,xdim,0,0,mlid);
//errstream() << "phantomxyz mer 1c " << yyprod << "\n";
                }
            }

            if ( ( size() >= 2 ) && needsDiff(1) && !justcalcip )
            {
                if ( pxyprod && pxyprod[1] )
                {
                    diffis = *pxyprod[1];
                }

                else
                {
                    T xnorm;
                    T ynorm;

                    int dummyind = 0;

//errstream() << "phantomxyz mer 2\n";
                    kernel8xx(0,xnorm,dummyind,cType(zeroint()),x,x,xinfo,xinfo,i,i,xdim,0,0,mlid);
//errstream() << "phantomxyz mer 2b" << xnorm << "\n";
                    kernel8xx(0,ynorm,dummyind,cType(zeroint()),y,y,yinfo,yinfo,j,j,xdim,0,0,mlid);
//errstream() << "phantomxyz mer 2c" << ynorm << "\n";

                    diff2norm(diffis,xyprod,xnorm,ynorm);
                }
            }

            if ( !justcalcip )
            {
                xyprod *= (const T &) cWeight(0);
                yxprod *= (const T &) cWeight(0);
                xxprod *= (const T &) cWeight(0);
                yyprod *= (const T &) cWeight(0);

                diffis *= (const T &) cWeight(0);
            }
        }

        if ( justcalcip )
        {
            res = xyprod;
        }

        else
        {
            if ( isNormalised(locindend) )
            {
//errstream() << "phantomxyz mer 3 cType = " << cType(zeroint()) << "\n";
//errstream() << "phantomxyz mer 3 xyprod = " << xyprod << "\n";
//errstream() << "phantomxyz mer 3 xxprod = " << xxprod << "\n";
//errstream() << "phantomxyz mer 3 yyprod = " << yyprod << "\n";
//errstream() << "phantomxyz mer 3 diffis = " << diffis << "\n";
                T dummy;

                T xyvals[2] = { xxprod,yyprod };
                T xxvals[2] = { xxprod,xxprod };
                T yyvals[2] = { yyprod,yyprod };

                int ixy[2] = { i,j };
                int ixx[2] = { i,i };
                int iyy[2] = { j,j };

                KKpro(res,xyprod,diffis,ixy,locindstart,locindend,xdim,2,dummy,xyvals);

                T xres;
                T yres;

                T zerodiff; zerodiff = 0.0;

                KKpro(xres,xxprod,zerodiff,ixx,locindstart,locindend,xdim,2,dummy,xxvals);
                KKpro(yres,yyprod,zerodiff,iyy,locindstart,locindend,xdim,2,dummy,yyvals);

                NiceAssert( !testisvnan(xres) );
                NiceAssert( !testisvnan(yres) );

                NiceAssert( !testisinf(xres) );
                NiceAssert( !testisinf(yres) );

//T xressave = xres;
//                xres *= yres;
//                OP_sqrt(xres);
//
//                NiceAssert( !testisvnan(xres) );
//                NiceAssert( !testisinf(xres) );
//
//T ressave = res;
//                safedivby(res,xres); //res /= xres;
//
//if ( testisinf(res) )
//{
//errstream() << "phantomx 0: res = " << ressave << "\n";
//errstream() << "phantomx 1: xres = " << xressave << "\n";
//errstream() << "phantomx 2: yres = " << yres << "\n";
//errstream() << "phantomx 3: sqrt(xres*yres) = " << xres << "\n";
//errstream() << "phantomx 4: res/sqrt(xres*yres) = " << res << "\n";
//}
//

// The above code tends to under/overflow with wild abandon.  So
// let's attack this with maths.  If xres or yres = 0 then, assuming
// a vaguely sane kernel, res must also be zero, so:
//
// res/sqrt(xres.yres) = exp(log(res)-(log(xres)-log(yres))/2)
// res/sqrt(xres.yres) = exp((2*log(res)-log(xres)-log(yres))/2)

static T zerores; zerores = 0.0;

if ( ( xres == zerores ) || ( yres == zerores ) )
{
    res = 1.0; // Correct up to unavoidable sign ambiguity
}

else
{
    T sgnres = angle(res);

    res = abs2(res);
    OP_log(res);
    res *= 2.0;
    res -= log(xres);
    res -= log(yres);
    res *= 0.5;
    OP_exp(res);
    res *= sgnres;
}

                NiceAssert( !testisvnan(res) );
                NiceAssert( !testisinf(res) );
            }

            else
            {
//errstream() << "phantomxyz mer 4 cType = " << cType(zeroint()) << "\n";
//errstream() << "phantomxyz mer 4 xyprod = " << xyprod << "\n";
//errstream() << "phantomxyz mer 4 diffis = " << diffis << "\n";
                T xyvals[2] = { xxprod,yyprod };
                int ixy[2] = { i,j };

                logresvalid = KKpro(res,xyprod,diffis,ixy,locindstart,locindend,xdim,2,logres,xyvals);
            }
        }
    }

    else
    {
        NiceAssert( ismagterm == zeroint() );
        NiceAssert( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) );

        gentype xyprod(0.0);
        gentype yxprod(0.0);

        gentype xnorm(0.0);
        gentype ynorm(0.0);

        if ( xy10 && xy00 && xy11)
        {
            xyprod = (*xy10);
            yxprod = (*xy10);

            xyprod += bias;
            yxprod += bias;

            xnorm = (*xy00);
            ynorm = (*xy11);
        }

        else if ( pxyprod && pxyprod[0] )
        {
            xyprod = *pxyprod[0];
            yxprod = xyprod;

            xnorm = getmnorm(xinfo,x,2,xconsist,assumreal);
            ynorm = getmnorm(yinfo,y,2,xconsist,assumreal);
        }

        else if ( needsInner(-1,2) )
        {
            // This may be used by some kernels and not others, so calculate anyhow

            twoProductDiverted(xyprod,x,y,xconsist,assumreal);
            twoProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

            xyprod += bias;
            yxprod += bias;

            xnorm = getmnorm(xinfo,x,2,xconsist,assumreal);
            ynorm = getmnorm(yinfo,y,2,xconsist,assumreal);
        }

        else
        {
            xnorm = getmnorm(xinfo,x,2,xconsist,assumreal);
            ynorm = getmnorm(yinfo,y,2,xconsist,assumreal);
        }

        gentype tempres;

        K2i(tempres,xyprod,yxprod,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,0,0,mlid,indstart,indend,assumreal);

        res = (T) tempres;
    }

    return res;
}

//phantomx
template <class T>
T &MercerKernel::LL3(T &res, T &logres, int &logresvalid,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                     const T &bias,
                     const gentype **pxyprod,
                     int ia, int ib, int ic, 
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, 
                     const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const Vector<int> *s, int justcalcip, int indstart, int indend) const
{
    NiceAssert( ! ( isFastKernelSum() && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) );

    logresvalid = 0;

    if ( !size() )
    {
        return res = 0.0;
    }

    if ( ( isAltDiff() == 300 ) && !justcalcip )
    {
        goto badout;
    }

    else if ( !s && ( isAltDiff() >= 100 ) && ( isAltDiff() <= 199 ) && !justcalcip )
    {
        goto badout;
    }

    else if ( ( resmode & 0x01 ) )
    {
        goto badout;
    }

    else if ( isprod && !arexysimple(xa,xb,xc) )
    {
        NiceAssert( !justcalcip );
        NiceAssert( !( resmode & 0x80 ) );

        // Should probably replace this with a proper fast version

        Vector<const SparseVector<gentype> *> xx(3);
        Vector<const vecInfo *> xxinfo(3);
        Vector<int> ii(3);

        xx("[]",zeroint()) = &xa;
        xx("[]",1)         = &xb;
        xx("[]",2)         = &xc;

        xxinfo("[]",zeroint()) = &xainfo;
        xxinfo("[]",1)         = &xbinfo;
        xxinfo("[]",2)         = &xcinfo;

        ii("[]",zeroint()) = ia;
        ii("[]",1)         = ib;
        ii("[]",2)         = ic;

        LLm(3,res,logres,logresvalid,xx,xxinfo,bias,ii,pxyprod,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,0,indstart,indend);
    }

    else if ( ( isFastKernelSum() || isFastKernelChain() ) && ( resmode && 0x80 ) && !justcalcip )
    {
        res = 0;
    }

    else if ( isFastKernelSum() || isFastKernelChain() || ( isFastKernelXfer() && !resmode ) || justcalcip )
    {
        int locindstart = (isFastKernelXfer()?1:indstart); // NB: isFastKernelXfer() implies indstart == 0
        int locindend   = indend; //size()-1;

        int needxxprod = isNormalised(locindend);

        T xyprod; xyprod = 0.0;
        T diffis; diffis = 0.0;
        T xaprod; xaprod = 0.0;
        T xbprod; xbprod = 0.0;
        T xcprod; xcprod = 0.0;

        // Calculate inner products and differences if required

        if ( isFastKernelSum() || isFastKernelChain() || justcalcip )
        {
            if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
            }

            else if ( needsInner(0,3) || ( isFastKernelSum() && needsInner(-1,3) ) || justcalcip )
            {
                threeProductDiverted(xyprod,xa,xb,xc,xconsist,assumreal);

                if ( !justcalcip )
                {
                    xyprod += bias;
                }
            }

            if ( needxxprod && !justcalcip )
            {
                threeProductDiverted(xaprod,xa,xa,xa,xconsist,assumreal);
                threeProductDiverted(xbprod,xb,xb,xb,xconsist,assumreal);
                threeProductDiverted(xcprod,xc,xc,xc,xconsist,assumreal);

                xaprod += bias;
                xbprod += bias;
                xcprod += bias;
            }

            if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else if ( ( ( needsDiff(0) || ( isFastKernelSum() && needsDiff() ) ) && ( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) ) ) && !justcalcip )
            {
                // Calculate ||x-y||^2 only as required

                double altxyr00 = 0;
                double altxyr10 = 0;
                double altxyr11 = 0;
                double altxyr20 = 0;
                double altxyr21 = 0;
                double altxyr22 = 0;

                fillXYMatrix(altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,xa,xb,xc,xainfo,xbinfo,xcinfo,xy00,xy10,xy11,xy20,xy21,xy22,0,assumreal);

                diff3norm(diffis,xyprod,getmnorm(xainfo,xa,3,xconsist,assumreal),getmnorm(xbinfo,xb,3,xconsist,assumreal),getmnorm(xcinfo,xc,3,xconsist,assumreal),altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,s);
            }

            else if ( ( needsDiff(0) && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) && !justcalcip )
            {
                goto badout;
            }
        }

        else
        {
            // ( isFastKernelXfer() && !resmode )
            // isSimpleFastKernelChain

            if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
            }

            else
            {
                int dummyind = 0;

                kernel8xx(0,xyprod,dummyind,cType(zeroint()),xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,0,0,mlid);
            }

            if ( needxxprod && !justcalcip )
            {
                int dummyind = 0;

                kernel8xx(0,xaprod,dummyind,cType(zeroint()),xa,xa,xa,xainfo,xainfo,xainfo,ia,ia,ia,xdim,0,0,mlid);
                kernel8xx(0,xbprod,dummyind,cType(zeroint()),xb,xb,xb,xbinfo,xbinfo,xbinfo,ib,ib,ib,xdim,0,0,mlid);
                kernel8xx(0,xcprod,dummyind,cType(zeroint()),xc,xc,xc,xcinfo,xcinfo,xcinfo,ic,ic,ic,xdim,0,0,mlid);
            }

            if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else if ( ( size() >= 2 ) && needsDiff(1) && ( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) ) && !justcalcip )
            {
                int dummyind = 0;

                T xanorm;
                T xbnorm;
                T xcnorm;

                double altxyr00 = 0;
                double altxyr10 = 0;
                double altxyr11 = 0;
                double altxyr20 = 0;
                double altxyr21 = 0;
                double altxyr22 = 0;

                if ( isAltDiff() == 0 )
                {
                    kernel8xx(0,xanorm,dummyind,cType(zeroint()),xa,xa,xa,xainfo,xainfo,xainfo,ia,ia,ia,xdim,0,0,mlid);
                    kernel8xx(0,xbnorm,dummyind,cType(zeroint()),xb,xb,xb,xbinfo,xbinfo,xbinfo,ib,ib,ib,xdim,0,0,mlid);
                    kernel8xx(0,xcnorm,dummyind,cType(zeroint()),xc,xc,xc,xcinfo,xcinfo,xcinfo,ic,ic,ic,xdim,0,0,mlid);

                    // needsMatDiff() == 0 here by definition
                }

                else
                {
                    kernel8xx(0,xanorm,dummyind,cType(zeroint()),xa,xa,xainfo,xainfo,ia,ia,xdim,0,0,mlid);
                    kernel8xx(0,xbnorm,dummyind,cType(zeroint()),xb,xb,xbinfo,xbinfo,ib,ib,xdim,0,0,mlid);
                    kernel8xx(0,xcnorm,dummyind,cType(zeroint()),xc,xc,xcinfo,xcinfo,ic,ic,xdim,0,0,mlid);

                    if ( needsMatDiff() )
                    {
                        kernel8xx(0,altxyr10,dummyind,cType(zeroint()),xb,xa,xbinfo,xainfo,ib,ia,xdim,0,0,mlid);
                        kernel8xx(0,altxyr20,dummyind,cType(zeroint()),xc,xa,xcinfo,xainfo,ic,ia,xdim,0,0,mlid);
                        kernel8xx(0,altxyr21,dummyind,cType(zeroint()),xc,xb,xcinfo,xbinfo,ic,ib,xdim,0,0,mlid);

                        altxyr00 = xanorm;
                        altxyr11 = xbnorm;
                        altxyr22 = xcnorm;
                    }
                }

                diff3norm(diffis,xyprod,xanorm,xbnorm,xcnorm,altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,s);

                if ( !justcalcip )
                {
                    xyprod *= (const T &) cWeight(0);

                    xaprod *= (const T &) cWeight(0);
                    xbprod *= (const T &) cWeight(0);
                    xcprod *= (const T &) cWeight(0);

                    diffis *= (const T &) cWeight(0);
                }
            }

            else if ( needsDiff(0) && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) && !justcalcip )
            {
                goto badout;
            }
        }

        if ( justcalcip )
        {
            res = xyprod;
        }

        else
        {
            if ( isNormalised(locindend) )
            {
throw("There is an error here.  diff != 0 in general");
                T dummy;

                int ixyz[3] = { ia,ib,ic };
                int ixxx[3] = { ia,ia,ia };
                int iyyy[3] = { ib,ib,ib };
                int izzz[3] = { ic,ic,ic };

                KKpro(res,xyprod,diffis,ixyz,locindstart,locindend,xdim,3,dummy,&dummy);

                T xares;
                T xbres;
                T xcres;

                T zerodiff; zerodiff = 0.0;

                KKpro(xares,xaprod,zerodiff,ixxx,locindstart,locindend,xdim,3,dummy,&dummy);
                KKpro(xbres,xbprod,zerodiff,iyyy,locindstart,locindend,xdim,3,dummy,&dummy);
                KKpro(xcres,xcprod,zerodiff,izzz,locindstart,locindend,xdim,3,dummy,&dummy);

                xares *= xbres;
                xares *= xcres;

                T oneonm; oneonm = 1.0/3.0;

                safedivby(res,pow(xares,oneonm)); //res /= pow(xares,oneonm);
            }

            else
            {
                T dummy;

                int ixyz[3] = { ia,ib,ic };

                logresvalid = KKpro(res,xyprod,diffis,ixyz,locindstart,locindend,xdim,3,logres,&dummy);
            }
        }
    }

    else
    {
        goto badout;
    }

    return res;

badout:
    // Design decision: in ml_base.cc, if d = 0 for one of the vectors
    // referenced here then this element will never be used.  Moreover there
    // are cases (eg isAltDiff set >1 with back-referenced data) where the
    // element is not properly defined but will never be used, so what we 
    // need to do is set it 0.  However having a "d = 0" catch will fail when
    // d starts non-zero, is set zero, then set non-zero, as will happen for
    // example when calculating LOO, n-fold error etc.  In such cases you need
    // to call a reset on that row/column, but you can't do that because (a) the
    // reset often calls setd (hence infinite recursion) or (b) there is an 
    // implicit assumption that Gp is independent of d (eg in semicopy functions
    // that retain the caches for speed in LOO, n-fold calculation).  Hence I've 
    // made the decision to return 0 here to avoid a whole stack of potential
    // coding complications at the price of possible silent failure if you set
    // somthing incorrectly.

    errstream() << "!!!3!!!";

    res = 0.0;

    return res;
}

//phantomx
template <class T>
T &MercerKernel::LL4(T &res, T &logres, int &logresvalid,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                     const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                     const T &bias,
                     const gentype **pxyprod,
                     int ia, int ib, int ic, int id, 
                     int xdim, int xconsist, int assumreal, int resmode, int mlid, 
                     const Vector<int> *s, int justcalcip,
                     const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int indstart, int indend) const
{
    NiceAssert( ! ( isFastKernelSum() && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) );
    NiceAssert( ( xy00 && xy10 && xy11 && xy20 && xy21 && xy22 && xy30 && xy31 && xy32 && xy33 ) || ( !xy00 && !xy10 && !xy11 && !xy20 && !xy21 && !xy22 && !xy30 && !xy31 && !xy32 && !xy33 ) );

    logresvalid = 0;

    T dummya;
    int dummyb = 0;

    if ( !size() )
    {
        return res = 0.0;
    }

    if ( ( isAltDiff() == 300 ) && !justcalcip )
    {
        NiceAssert( !s );

        T tempresa; tempresa = 0.0;
        T tempresb; tempresb = 0.0;

        res = 0.0;

        LL2(tempresa,dummya,dummyb,xa,xb,xainfo,xbinfo,bias,NULL,ia,ib,xdim,xconsist,assumreal,resmode,mlid,xy00,xy10,xy11,justcalcip,indstart,indend); 
        LL2(tempresb,dummya,dummyb,xc,xd,xcinfo,xdinfo,bias,NULL,ic,id,xdim,xconsist,assumreal,resmode,mlid,xy22,xy32,xy33,justcalcip,indstart,indend); 

        tempresa *= tempresb; 
        res += tempresa;

        LL2(tempresa,dummya,dummyb,xa,xc,xainfo,xcinfo,bias,NULL,ia,ic,xdim,xconsist,assumreal,resmode,mlid,xy00,xy20,xy22,justcalcip,indstart,indend); 
        LL2(tempresb,dummya,dummyb,xb,xd,xcinfo,xdinfo,bias,NULL,ib,id,xdim,xconsist,assumreal,resmode,mlid,xy11,xy31,xy33,justcalcip,indstart,indend); 

        tempresa *= tempresb; 
        res += tempresa;

        LL2(tempresa,dummya,dummyb,xa,xd,xainfo,xdinfo,bias,NULL,ia,id,xdim,xconsist,assumreal,resmode,mlid,xy00,xy30,xy33,justcalcip,indstart,indend); 
        LL2(tempresb,dummya,dummyb,xb,xc,xbinfo,xcinfo,bias,NULL,ib,ic,xdim,xconsist,assumreal,resmode,mlid,xy11,xy21,xy22,justcalcip,indstart,indend); 

        tempresa *= tempresb; 
        res += tempresa;

        res /= 3;
    }

    else if ( !s && ( isAltDiff() >= 100 ) && ( isAltDiff() <= 199 ) && !justcalcip )
    {
        Vector<int> ss(4);
        T tempres; tempres = 0.0;
        int z = 0;

        res = 0.0;

        if ( isAltDiff() == 103 )
        {
            ss("&",z) = +1; ss("&",1) = +1; ss("&",2) = +1; ss("&",3) = +1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = +1; ss("&",1) = +1; ss("&",2) = -1; ss("&",3) = -1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = +1; ss("&",1) = -1; ss("&",2) = +1; ss("&",3) = -1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = +1; ss("&",1) = -1; ss("&",2) = -1; ss("&",3) = +1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;

            ss("&",z) = -1; ss("&",1) = -1; ss("&",2) = -1; ss("&",3) = -1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = -1; ss("&",1) = -1; ss("&",2) = +1; ss("&",3) = +1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = -1; ss("&",1) = +1; ss("&",2) = -1; ss("&",3) = +1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = -1; ss("&",1) = +1; ss("&",2) = +1; ss("&",3) = -1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;

            res /= 8;
        }

        else if ( isAltDiff() == 104 )
        {
            ss("&",z) = 0; ss("&",1) = 1; ss("&",2) = 2; ss("&",3) = 3; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 0; ss("&",1) = 1; ss("&",2) = 3; ss("&",3) = 2; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 0; ss("&",1) = 2; ss("&",2) = 1; ss("&",3) = 3; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 0; ss("&",1) = 2; ss("&",2) = 3; ss("&",3) = 1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 0; ss("&",1) = 3; ss("&",2) = 1; ss("&",3) = 2; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 0; ss("&",1) = 3; ss("&",2) = 2; ss("&",3) = 1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;

            ss("&",z) = 1; ss("&",1) = 0; ss("&",2) = 2; ss("&",3) = 3; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 1; ss("&",1) = 0; ss("&",2) = 3; ss("&",3) = 2; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 1; ss("&",1) = 2; ss("&",2) = 0; ss("&",3) = 3; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 1; ss("&",1) = 2; ss("&",2) = 3; ss("&",3) = 0; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 1; ss("&",1) = 3; ss("&",2) = 0; ss("&",3) = 2; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 1; ss("&",1) = 3; ss("&",2) = 2; ss("&",3) = 0; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;

            ss("&",z) = 2; ss("&",1) = 1; ss("&",2) = 0; ss("&",3) = 3; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 2; ss("&",1) = 1; ss("&",2) = 3; ss("&",3) = 0; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 2; ss("&",1) = 0; ss("&",2) = 1; ss("&",3) = 3; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 2; ss("&",1) = 0; ss("&",2) = 3; ss("&",3) = 1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 2; ss("&",1) = 3; ss("&",2) = 1; ss("&",3) = 0; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 2; ss("&",1) = 3; ss("&",2) = 0; ss("&",3) = 1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;

            ss("&",z) = 3; ss("&",1) = 1; ss("&",2) = 2; ss("&",3) = 0; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 3; ss("&",1) = 1; ss("&",2) = 0; ss("&",3) = 2; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 3; ss("&",1) = 2; ss("&",2) = 1; ss("&",3) = 0; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 3; ss("&",1) = 2; ss("&",2) = 0; ss("&",3) = 1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 3; ss("&",1) = 0; ss("&",2) = 1; ss("&",3) = 2; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;
            ss("&",z) = 3; ss("&",1) = 0; ss("&",2) = 2; ss("&",3) = 1; LL4(tempres,dummya,dummyb,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,NULL,ia,ib,ic,id,xdim,xconsist,assumreal,resmode,mlid,&ss,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,indstart,indend); res += tempres;

            res /= 24;
        }
    }

    else if ( ( resmode & 0x01 ) )
    {
        NiceAssert( !justcalcip );
        NiceAssert( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) );
        NiceAssert( !( isprod && !arexysimple(xa,xb) && !arexysimple(xa,xc) && !arexysimple(xa,xd) ) );

        double altxyr00 = 0;
        double altxyr10 = 0;
        double altxyr11 = 0;
        double altxyr20 = 0;
        double altxyr21 = 0;
        double altxyr22 = 0;
        double altxyr30 = 0;
        double altxyr31 = 0;
        double altxyr32 = 0;
        double altxyr33 = 0;

        fillXYMatrix(altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,altxyr30,altxyr31,altxyr32,altxyr33,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,0,assumreal);

        gentype tempres;

        K4i(tempres,defaultgentype(),xainfo,xbinfo,xcinfo,xdinfo,getmnorm(xainfo,xa,4,xconsist,assumreal),getmnorm(xbinfo,xb,4,xconsist,assumreal),getmnorm(xcinfo,xc,4,xconsist,assumreal),getmnorm(xdinfo,xd,4,xconsist,assumreal),xa,xb,xc,xd,ia,ib,ic,id,xdim,0,resmode,mlid,altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,altxyr30,altxyr31,altxyr32,altxyr33,s,indstart,indend,assumreal);

        res = (T) tempres;
    }

    else if ( isprod && !arexysimple(xa,xb,xc,xd) )
    {
        NiceAssert( !justcalcip );
        NiceAssert( !( resmode & 0x80 ) );

        // Should probably replace this with a proper fast version

        Vector<const SparseVector<gentype> *> xx(4);
        Vector<const vecInfo *> xxinfo(4);
        Vector<int> ii(4);

        xx("[]",zeroint()) = &xa;
        xx("[]",1)         = &xb;
        xx("[]",2)         = &xc;
        xx("[]",3)         = &xd;

        xxinfo("[]",zeroint()) = &xainfo;
        xxinfo("[]",1)         = &xbinfo;
        xxinfo("[]",2)         = &xcinfo;
        xxinfo("[]",3)         = &xdinfo;

        ii("[]",zeroint()) = ia;
        ii("[]",1)         = ib;
        ii("[]",2)         = ic;
        ii("[]",3)         = id;

        LLm(4,res,logres,logresvalid,xx,xxinfo,bias,ii,pxyprod,xdim,xconsist,assumreal,resmode,mlid,NULL,NULL,0,indstart,indend);
    }

    else if ( ( isFastKernelSum() || isFastKernelChain() ) && ( resmode && 0x80 ) && !justcalcip )
    {
        res = 0;
    }

    else if ( isFastKernelSum() || isFastKernelChain() || ( isFastKernelXfer() && !resmode ) || justcalcip )
    {
        int locindstart = (isFastKernelXfer()?1:indstart); // NB: isFastKernelXfer() implies indstart == 0
        int locindend   = indend; //size()-1;

        int needxxprod = isNormalised(locindend);

        T xyprod; xyprod = 0.0;
        T diffis; diffis = 0.0;
        T xaprod; xaprod = 0.0;
        T xbprod; xbprod = 0.0;
        T xcprod; xcprod = 0.0;
        T xdprod; xdprod = 0.0;

        // Calculate inner products and differences if required

        if ( isFastKernelSum() || isFastKernelChain() || justcalcip )
        {
            if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
            }

            else if ( needsInner(0,4) || ( isFastKernelSum() && needsInner(-1,4) ) || justcalcip )
            {
                fourProductDiverted(xyprod,xa,xb,xc,xd,xconsist,assumreal);

                if ( !justcalcip )
                {
                    xyprod += bias;
                }
            }

            if ( needxxprod && !justcalcip )
            {
                fourProductDiverted(xaprod,xa,xa,xa,xa,xconsist,assumreal);
                fourProductDiverted(xbprod,xb,xb,xb,xb,xconsist,assumreal);
                fourProductDiverted(xcprod,xc,xc,xc,xc,xconsist,assumreal);
                fourProductDiverted(xdprod,xd,xd,xd,xd,xconsist,assumreal);

                xaprod += bias;
                xbprod += bias;
                xcprod += bias;
                xdprod += bias;
            }

            if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else if ( ( ( needsDiff(0) || ( isFastKernelSum() && needsDiff() ) ) && ( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) ) ) && !justcalcip )
            {
                // Calculate ||x-y||^2 only as required

                double altxyr00 = 0;
                double altxyr10 = 0;
                double altxyr11 = 0;
                double altxyr20 = 0;
                double altxyr21 = 0;
                double altxyr22 = 0;
                double altxyr30 = 0;
                double altxyr31 = 0;
                double altxyr32 = 0;
                double altxyr33 = 0;

                fillXYMatrix(altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,altxyr30,altxyr31,altxyr32,altxyr33,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,0,assumreal);

                diff4norm(diffis,xyprod,getmnorm(xainfo,xa,4,xconsist,assumreal),getmnorm(xbinfo,xb,4,xconsist,assumreal),getmnorm(xcinfo,xc,4,xconsist,assumreal),getmnorm(xdinfo,xd,4,xconsist,assumreal),altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,altxyr30,altxyr31,altxyr32,altxyr33,s);
            }

            else if ( ( needsDiff(0) && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) && !justcalcip )
            {
                NiceAssert( !needxxprod );

                // At this point we need to calculate diffis using altdiffis method 2xx
                // xyprod is not used by kernel, but need to fill it in for use by rest of chain
                // we only need to cycle through diffis for relevant s vectors
                // We need the xy matrix to do this

                int z = 0;
                T tempres;

                double ssxyzz = 0;
                double ssxy1z = 0;
                double ssxy11 = 0;
                double ssxy2z = 0;
                double ssxy21 = 0;
                double ssxy22 = 0;
                double ssxy3z = 0;
                double ssxy31 = 0;
                double ssxy32 = 0;
                double ssxy33 = 0;

                double &ssxyz1 = ssxy1z;
                double &ssxyz2 = ssxy2z;
                double &ssxyz3 = ssxy3z;
                double &ssxy12 = ssxy21;
                double &ssxy13 = ssxy31;
                double &ssxy23 = ssxy32;

                fillXYMatrix(ssxyzz,ssxy1z,ssxy11,ssxy2z,ssxy21,ssxy22,ssxy3z,ssxy31,ssxy32,ssxy33,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,1,assumreal);

                xyprod = 0.0;

                if ( isAltDiff() == 203 )
                {
                    T dummy;

                    int ii[4] = { ia,ib,ic,id };

                    // ++--

                    diffis  =  ssxyzz + ssxyz1 - ssxyz2 - ssxyz3; 
                    diffis +=  ssxy1z + ssxy11 - ssxy12 - ssxy13; 
                    diffis += -ssxy2z - ssxy21 + ssxy22 + ssxy23; 
                    diffis += -ssxy3z - ssxy31 + ssxy32 + ssxy33; 

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // +-+-

                    diffis  =  ssxyzz - ssxyz1 + ssxyz2 - ssxyz3;
                    diffis += -ssxy1z + ssxy11 - ssxy12 + ssxy13;
                    diffis +=  ssxy2z - ssxy21 + ssxy22 - ssxy23;
                    diffis += -ssxy3z + ssxy31 - ssxy32 + ssxy33;

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // +--+

                    diffis  =  ssxyzz - ssxyz1 - ssxyz2 + ssxyz3;
                    diffis += -ssxy1z + ssxy11 + ssxy12 - ssxy13;
                    diffis += -ssxy2z + ssxy21 + ssxy22 - ssxy23;
                    diffis +=  ssxy3z - ssxy31 - ssxy32 + ssxy33;

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // --++

                    diffis  =  ssxyzz + ssxyz1 - ssxyz2 - ssxyz3;
                    diffis +=  ssxy1z + ssxy11 - ssxy12 - ssxy13;
                    diffis += -ssxy2z - ssxy21 + ssxy22 + ssxy23;
                    diffis += -ssxy3z - ssxy31 + ssxy32 + ssxy33;

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // -+-+

                    diffis  =  ssxyzz - ssxyz1 + ssxyz2 - ssxyz3;
                    diffis += -ssxy1z + ssxy11 - ssxy12 + ssxy13;
                    diffis +=  ssxy2z - ssxy21 + ssxy22 - ssxy23;
                    diffis += -ssxy3z + ssxy31 - ssxy32 + ssxy33;

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // -++-

                    diffis  =  ssxyzz - ssxyz1 - ssxyz2 + ssxyz3;
                    diffis += -ssxy1z + ssxy11 + ssxy12 - ssxy13;
                    diffis += -ssxy2z + ssxy21 + ssxy22 - ssxy23;
                    diffis +=  ssxy3z - ssxy31 - ssxy32 + ssxy33;

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // ++++

                    diffis  =  ssxyzz + ssxyz1 + ssxyz2 + ssxyz3;
                    diffis +=  ssxy1z + ssxy11 + ssxy12 + ssxy13;
                    diffis +=  ssxy2z + ssxy21 + ssxy22 + ssxy23;
                    diffis +=  ssxy3z + ssxy31 + ssxy32 + ssxy33;

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // ----

                    diffis  =  ssxyzz + ssxyz1 + ssxyz2 + ssxyz3;
                    diffis +=  ssxy1z + ssxy11 + ssxy12 + ssxy13;
                    diffis +=  ssxy2z + ssxy21 + ssxy22 + ssxy23;
                    diffis +=  ssxy3z + ssxy31 + ssxy32 + ssxy33;

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    xyprod /= 8;
                }

                else if ( isAltDiff() == 204 )
                {
                    Matrix<double> ssxy;

                    ssxy.resize(4,4);

                    ssxy("&",z,z) = ssxyzz; ssxy("&",z,1) = ssxyz1; ssxy("&",z,2) = ssxyz2; ssxy("&",z,3) = ssxyz3;
                    ssxy("&",1,z) = ssxy1z; ssxy("&",1,1) = ssxy11; ssxy("&",1,2) = ssxy12; ssxy("&",1,3) = ssxy13;
                    ssxy("&",2,z) = ssxy2z; ssxy("&",2,1) = ssxy21; ssxy("&",2,2) = ssxy22; ssxy("&",2,3) = ssxy23;
                    ssxy("&",3,z) = ssxy3z; ssxy("&",3,1) = ssxy31; ssxy("&",3,2) = ssxy32; ssxy("&",3,3) = ssxy33;

                    T dummy;

                    int ii[4] = { ia,ib,ic,id };

                    Vector<int> ss(4);

                    ss("&",z) = 0; ss("&",1) = 1; ss("&",2) = 2; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 1; ss("&",2) = 3; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 2; ss("&",2) = 1; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 2; ss("&",2) = 3; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 3; ss("&",2) = 1; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 3; ss("&",2) = 2; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;


                    ss("&",z) = 1; ss("&",1) = 0; ss("&",2) = 2; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 0; ss("&",2) = 3; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 2; ss("&",2) = 0; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 2; ss("&",2) = 3; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 3; ss("&",2) = 0; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 3; ss("&",2) = 2; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;


                    ss("&",z) = 2; ss("&",1) = 1; ss("&",2) = 0; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 1; ss("&",2) = 3; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 0; ss("&",2) = 1; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 0; ss("&",2) = 3; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 3; ss("&",2) = 1; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 3; ss("&",2) = 0; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;


                    ss("&",z) = 3; ss("&",1) = 1; ss("&",2) = 2; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 1; ss("&",2) = 0; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 2; ss("&",2) = 1; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 2; ss("&",2) = 0; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 0; ss("&",2) = 1; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 0; ss("&",2) = 2; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;


                    xyprod /= 24;
                }

                locindstart++;
            }
        }

        else
        {
            // ( isFastKernelXfer() && !resmode )
            // isSimpleFastKernelChain

            if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
            }

            else
            {
                int dummyind = 0;

                kernel8xx(0,xyprod,dummyind,cType(zeroint()),xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,0,0,mlid);
            }

            if ( needxxprod && !justcalcip )
            {
                int dummyind = 0;

                kernel8xx(0,xaprod,dummyind,cType(zeroint()),xa,xa,xa,xa,xainfo,xainfo,xainfo,xainfo,ia,ia,ia,ia,xdim,0,0,mlid);
                kernel8xx(0,xbprod,dummyind,cType(zeroint()),xb,xb,xb,xb,xbinfo,xbinfo,xbinfo,xbinfo,ib,ib,ib,ib,xdim,0,0,mlid);
                kernel8xx(0,xcprod,dummyind,cType(zeroint()),xc,xc,xc,xc,xcinfo,xcinfo,xcinfo,xcinfo,ic,ic,ic,ic,xdim,0,0,mlid);
                kernel8xx(0,xdprod,dummyind,cType(zeroint()),xd,xd,xd,xd,xdinfo,xdinfo,xdinfo,xdinfo,id,id,id,id,xdim,0,0,mlid);
            }

            if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else if ( ( size() >= 2 ) && needsDiff(1) && ( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) ) && !justcalcip )
            {
                int dummyind = 0;

                T xanorm;
                T xbnorm;
                T xcnorm;
                T xdnorm;

                double altxyr00 = 0;
                double altxyr10 = 0;
                double altxyr11 = 0;
                double altxyr20 = 0;
                double altxyr21 = 0;
                double altxyr22 = 0;
                double altxyr30 = 0;
                double altxyr31 = 0;
                double altxyr32 = 0;
                double altxyr33 = 0;

                if ( isAltDiff() == 0 )
                {
                    kernel8xx(0,xanorm,dummyind,cType(zeroint()),xa,xa,xa,xa,xainfo,xainfo,xainfo,xainfo,ia,ia,ia,ia,xdim,0,0,mlid);
                    kernel8xx(0,xbnorm,dummyind,cType(zeroint()),xb,xb,xb,xb,xbinfo,xbinfo,xbinfo,xbinfo,ib,ib,ib,ib,xdim,0,0,mlid);
                    kernel8xx(0,xcnorm,dummyind,cType(zeroint()),xc,xc,xc,xc,xcinfo,xcinfo,xcinfo,xcinfo,ic,ic,ic,ic,xdim,0,0,mlid);
                    kernel8xx(0,xdnorm,dummyind,cType(zeroint()),xd,xd,xd,xd,xdinfo,xdinfo,xdinfo,xdinfo,id,id,id,id,xdim,0,0,mlid);

                    // needsMatDiff() == 0 here by definition
                }

                else
                {
                    kernel8xx(0,xanorm,dummyind,cType(zeroint()),xa,xa,xainfo,xainfo,ia,ia,xdim,0,0,mlid);
                    kernel8xx(0,xbnorm,dummyind,cType(zeroint()),xb,xb,xbinfo,xbinfo,ib,ib,xdim,0,0,mlid);
                    kernel8xx(0,xcnorm,dummyind,cType(zeroint()),xc,xc,xcinfo,xcinfo,ic,ic,xdim,0,0,mlid);
                    kernel8xx(0,xdnorm,dummyind,cType(zeroint()),xd,xd,xdinfo,xdinfo,id,id,xdim,0,0,mlid);

                    if ( needsMatDiff() )
                    {
                        kernel8xx(0,altxyr10,dummyind,cType(zeroint()),xb,xa,xbinfo,xainfo,ib,ia,xdim,0,0,mlid);
                        kernel8xx(0,altxyr20,dummyind,cType(zeroint()),xc,xa,xcinfo,xainfo,ic,ia,xdim,0,0,mlid);
                        kernel8xx(0,altxyr21,dummyind,cType(zeroint()),xc,xb,xcinfo,xbinfo,ic,ib,xdim,0,0,mlid);
                        kernel8xx(0,altxyr30,dummyind,cType(zeroint()),xd,xa,xdinfo,xainfo,id,ia,xdim,0,0,mlid);
                        kernel8xx(0,altxyr31,dummyind,cType(zeroint()),xd,xb,xdinfo,xbinfo,id,ib,xdim,0,0,mlid);
                        kernel8xx(0,altxyr32,dummyind,cType(zeroint()),xd,xc,xdinfo,xcinfo,id,ic,xdim,0,0,mlid);

                        altxyr00 = xanorm;
                        altxyr11 = xbnorm;
                        altxyr22 = xcnorm;
                        altxyr33 = xdnorm;
                    }
                }

                diff4norm(diffis,xyprod,xanorm,xbnorm,xcnorm,xdnorm,altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,altxyr30,altxyr31,altxyr32,altxyr33,s);

                if ( !justcalcip )
                {
                    xyprod *= (const T &) cWeight(0);

                    xaprod *= (const T &) cWeight(0);
                    xbprod *= (const T &) cWeight(0);
                    xcprod *= (const T &) cWeight(0);
                    xdprod *= (const T &) cWeight(0);

                    diffis *= (const T &) cWeight(0);
                }
            }

            else if ( needsDiff(0) && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) && !justcalcip )
            {
                // At this point we need to calculate diffis using altdiffis method 2xx
                // xyprod is not used by kernel
                // we only need to cycle through diffis for relevant s vectors
                // We need the xy matrix to do this

                int z = 0;
                T tempres; tempres = 0.0;
                int dummyind = 0;

                Matrix<double> altxy(4,4);
                const Matrix<double> &ssxy = altxy;

                kernel8xx(0,altxy("&",z,z),dummyind,cType(zeroint()),xa,xa,xainfo,xainfo,ia,ia,xdim,0,0,mlid);
                kernel8xx(0,altxy("&",z,1),dummyind,cType(zeroint()),xa,xb,xainfo,xbinfo,ia,ib,xdim,0,0,mlid);
                kernel8xx(0,altxy("&",z,2),dummyind,cType(zeroint()),xa,xc,xainfo,xcinfo,ia,ic,xdim,0,0,mlid);
                kernel8xx(0,altxy("&",z,3),dummyind,cType(zeroint()),xa,xd,xainfo,xdinfo,ia,id,xdim,0,0,mlid);
                kernel8xx(0,altxy("&",1,1),dummyind,cType(zeroint()),xb,xb,xbinfo,xbinfo,ib,ib,xdim,0,0,mlid);
                kernel8xx(0,altxy("&",1,2),dummyind,cType(zeroint()),xb,xc,xbinfo,xcinfo,ib,ic,xdim,0,0,mlid);
                kernel8xx(0,altxy("&",1,3),dummyind,cType(zeroint()),xb,xd,xbinfo,xdinfo,ib,id,xdim,0,0,mlid);
                kernel8xx(0,altxy("&",2,2),dummyind,cType(zeroint()),xc,xc,xcinfo,xcinfo,ic,ic,xdim,0,0,mlid);
                kernel8xx(0,altxy("&",2,3),dummyind,cType(zeroint()),xc,xd,xcinfo,xdinfo,ic,id,xdim,0,0,mlid);
                kernel8xx(0,altxy("&",3,3),dummyind,cType(zeroint()),xd,xd,xdinfo,xdinfo,id,id,xdim,0,0,mlid);

                altxy("&",1,z) = altxy(z,1);
                altxy("&",2,z) = altxy(z,2);
                altxy("&",2,1) = altxy(1,2);
                altxy("&",3,z) = altxy(z,3);
                altxy("&",3,1) = altxy(1,3);
                altxy("&",3,2) = altxy(2,3);

                xyprod = 0.0;

                if ( isAltDiff() == 203 )
                {
                    T dummy;

                    int ii[4] = { ia,ib,ic,id };

                    // ++--

                    diffis  =  ssxy(z,z) + ssxy(z,1) - ssxy(z,2) - ssxy(z,3); 
                    diffis +=  ssxy(1,z) + ssxy(1,1) - ssxy(1,2) - ssxy(1,3); 
                    diffis += -ssxy(2,z) - ssxy(2,1) + ssxy(2,2) + ssxy(2,3); 
                    diffis += -ssxy(3,z) - ssxy(3,1) + ssxy(3,2) + ssxy(3,3); 

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // +-+-

                    diffis  =  ssxy(z,z) - ssxy(z,1) + ssxy(z,2) - ssxy(z,3);
                    diffis += -ssxy(1,z) + ssxy(1,1) - ssxy(1,2) + ssxy(1,3);
                    diffis +=  ssxy(2,z) - ssxy(2,1) + ssxy(2,2) - ssxy(2,3);
                    diffis += -ssxy(3,z) + ssxy(3,1) - ssxy(3,2) + ssxy(3,3);

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // +--+

                    diffis  =  ssxy(z,z) - ssxy(z,1) - ssxy(z,2) + ssxy(z,3);
                    diffis += -ssxy(1,z) + ssxy(1,1) + ssxy(1,2) - ssxy(1,3);
                    diffis += -ssxy(2,z) + ssxy(2,1) + ssxy(2,2) - ssxy(2,3);
                    diffis +=  ssxy(3,z) - ssxy(3,1) - ssxy(3,2) + ssxy(3,3);

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // --++

                    diffis  =  ssxy(z,z) + ssxy(z,1) - ssxy(z,2) - ssxy(z,3);
                    diffis +=  ssxy(1,z) + ssxy(1,1) - ssxy(1,2) - ssxy(1,3);
                    diffis += -ssxy(2,z) - ssxy(2,1) + ssxy(2,2) + ssxy(2,3);
                    diffis += -ssxy(3,z) - ssxy(3,1) + ssxy(3,2) + ssxy(3,3);

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // -+-+

                    diffis  =  ssxy(z,z) - ssxy(z,1) + ssxy(z,2) - ssxy(z,3);
                    diffis += -ssxy(1,z) + ssxy(1,1) - ssxy(1,2) + ssxy(1,3);
                    diffis +=  ssxy(2,z) - ssxy(2,1) + ssxy(2,2) - ssxy(2,3);
                    diffis += -ssxy(3,z) + ssxy(3,1) - ssxy(3,2) + ssxy(3,3);

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // -++-

                    diffis  =  ssxy(z,z) - ssxy(z,1) - ssxy(z,2) + ssxy(z,3);
                    diffis += -ssxy(1,z) + ssxy(1,1) + ssxy(1,2) - ssxy(1,3);
                    diffis += -ssxy(2,z) + ssxy(2,1) + ssxy(2,2) - ssxy(2,3);
                    diffis +=  ssxy(3,z) - ssxy(3,1) - ssxy(3,2) + ssxy(3,3);

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    // ++++

                    diffis  =  ssxy(z,z) + ssxy(z,1) + ssxy(z,2) + ssxy(z,3);
                    diffis +=  ssxy(1,z) + ssxy(1,1) + ssxy(1,2) + ssxy(1,3);
                    diffis +=  ssxy(2,z) + ssxy(2,1) + ssxy(2,2) + ssxy(2,3);
                    diffis +=  ssxy(3,z) + ssxy(3,1) + ssxy(3,2) + ssxy(3,3);

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;
	
                    // ----

                    diffis  =  ssxy(z,z) + ssxy(z,1) + ssxy(z,2) + ssxy(z,3);
                    diffis +=  ssxy(1,z) + ssxy(1,1) + ssxy(1,2) + ssxy(1,3);
                    diffis +=  ssxy(2,z) + ssxy(2,1) + ssxy(2,2) + ssxy(2,3);
                    diffis +=  ssxy(3,z) + ssxy(3,1) + ssxy(3,2) + ssxy(3,3);

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    xyprod /= 8;
                }

                else if ( isAltDiff() == 204 )
                {
                    T dummy;

                    int ii[4] = { ia,ib,ic,id };

                    Vector<int> ss(4);

                    ss("&",z) = 0; ss("&",1) = 1; ss("&",2) = 2; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 1; ss("&",2) = 3; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 2; ss("&",2) = 1; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 2; ss("&",2) = 3; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 3; ss("&",2) = 1; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 0; ss("&",1) = 3; ss("&",2) = 2; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;


                    ss("&",z) = 1; ss("&",1) = 0; ss("&",2) = 2; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 0; ss("&",2) = 3; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 2; ss("&",2) = 0; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 2; ss("&",2) = 3; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 3; ss("&",2) = 0; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 1; ss("&",1) = 3; ss("&",2) = 2; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;


                    ss("&",z) = 2; ss("&",1) = 1; ss("&",2) = 0; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 1; ss("&",2) = 3; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 0; ss("&",2) = 1; ss("&",3) = 3;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 0; ss("&",2) = 3; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 3; ss("&",2) = 1; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 2; ss("&",1) = 3; ss("&",2) = 0; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;


                    ss("&",z) = 3; ss("&",1) = 1; ss("&",2) = 2; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 1; ss("&",2) = 0; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 2; ss("&",2) = 1; ss("&",3) = 0;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 2; ss("&",2) = 0; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 0; ss("&",2) = 1; ss("&",3) = 2;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;

                    ss("&",z) = 3; ss("&",1) = 0; ss("&",2) = 2; ss("&",3) = 1;

                    diffis =  ssxy(ss(z),ss(z)) - ssxy(ss(z),ss(1));
                    diffis = -ssxy(ss(1),ss(z)) - ssxy(ss(1),ss(1));

                    diffis =  ssxy(ss(2),ss(2)) - ssxy(ss(2),ss(3));
                    diffis = -ssxy(ss(3),ss(2)) - ssxy(ss(3),ss(3));

                    KKpro(tempres,xyprod,diffis,ii,locindstart,locindstart,xdim,4,dummy,&dummy);
                    xyprod += tempres;


                    xyprod /= 24;
                }

                diffis *= (const T &) cWeight(0);

                locindstart++;
            }
        }

        if ( justcalcip )
        {
            res = xyprod;
        }

        else
        {
            if ( isNormalised(locindend) )
            {
throw("There's another error here: diff != 0 in general\n");
                T dummy;

                int iabcd[4] = { ia,ib,ic,id };
                int iaaaa[4] = { ia,ia,ia,ia };
                int ibbbb[4] = { ib,ib,ib,ib };
                int icccc[4] = { ic,ic,ic,ic };
                int idddd[4] = { id,id,id,id };

                KKpro(res,xyprod,diffis,iabcd,locindstart,locindend,xdim,4,dummy,&dummy);

                T xares;
                T xbres;
                T xcres;
                T xdres;

                T zerodiff; zerodiff = 0.0;

                KKpro(xares,xaprod,zerodiff,iaaaa,locindstart,locindend,xdim,4,dummy,&dummy);
                KKpro(xbres,xbprod,zerodiff,ibbbb,locindstart,locindend,xdim,4,dummy,&dummy);
                KKpro(xcres,xcprod,zerodiff,icccc,locindstart,locindend,xdim,4,dummy,&dummy);
                KKpro(xdres,xdprod,zerodiff,idddd,locindstart,locindend,xdim,4,dummy,&dummy);

                xares *= xbres;
                xares *= xcres;
                xares *= xdres;

                OP_sqrt(xares);
                OP_sqrt(xares);

                safedivby(res,xares); //res /= xares;
            }

            else
            {
                T dummy;

                int iabcd[4] = { ia,ib,ic,id };

                logresvalid = KKpro(res,xyprod,diffis,iabcd,locindstart,locindend,xdim,4,logres,&dummy);
            }
        }
    }

    else
    {
        NiceAssert( ismagterm == zeroint() );
        NiceAssert( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) );

        gentype xyprod(0.0);

        if ( pxyprod && pxyprod[0] )
        {
            xyprod = *pxyprod[0];
        }

        else if ( needsInner(-1,4) )
        {
            // This may be used by some kernels and not others, so calculate anyhow

            fourProductDiverted(xyprod,xa,xb,xc,xd,xconsist,assumreal);

            xyprod += bias;
        }

        double altxyr00 = 0;
        double altxyr10 = 0;
        double altxyr11 = 0;
        double altxyr20 = 0;
        double altxyr21 = 0;
        double altxyr22 = 0;
        double altxyr30 = 0;
        double altxyr31 = 0;
        double altxyr32 = 0;
        double altxyr33 = 0;

        fillXYMatrix(altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,altxyr30,altxyr31,altxyr32,altxyr33,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,0,assumreal);

        gentype tempres;

        K4i(tempres,xyprod,xainfo,xbinfo,xcinfo,xdinfo,getmnorm(xainfo,xa,4,xconsist,assumreal),getmnorm(xbinfo,xb,4,xconsist,assumreal),getmnorm(xcinfo,xc,4,xconsist,assumreal),getmnorm(xdinfo,xd,4,xconsist,assumreal),xa,xb,xc,xd,ia,ib,ic,id,xdim,0,resmode,mlid,altxyr00,altxyr10,altxyr11,altxyr20,altxyr21,altxyr22,altxyr30,altxyr31,altxyr32,altxyr33,s,indstart,indend,assumreal);

        res = (T) tempres;
    }

    return res;
}

//phantomx
template <class T>
T &MercerKernel::LLm(int m, T &res, T &logres, int &logresvalid,
                     Vector<const SparseVector<gentype> *> &x,
                     Vector<const vecInfo *> &xinfo,
                     const T &bias,
                     Vector<int> &iv,
                     const gentype **pxyprod, int xdim, int xconsist, int assumreal, int resmode, int mlid, 
                     const Matrix<double> *xy, const Vector<int> *s, int justcalcip, int indstart, int indend) const
{
    Vector<int> i(iv);

    NiceAssert( ! ( isFastKernelSum() && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 300 ) ) );

    logresvalid = 0;

    T dummya;
    int dummyb = 0;

    if ( (m%2) && ( isAltDiff() >= 100 ) )
    {
        goto badout;
    }

    if ( !size() )
    {
        return res = 0.0;
    }

    if ( isAltDiff() == 300 )
    {
        throw("I should probably implement this someday");
    }

    else if ( !s && ( isAltDiff() >= 100 ) && ( isAltDiff() <= 199 ) && !justcalcip )
    {
        Vector<int> ss(m);
        T tempres; tempres = 0.0;
        int ii,jj;

        res = 0.0;

        if ( isAltDiff() == 103 )
        {
            int isdone = 0;

            ss = 1;

            while ( !isdone )
            {
                if ( sum(ss)%4 == 0 )
                {
                    LLm(m,tempres,dummya,dummyb,x,xinfo,bias,i,NULL,xdim,xconsist,assumreal,resmode,mlid,xy,&ss,0,indstart,indend);
                    res += tempres;
                }

                isdone = 1;
                ii = 0;

                while ( ( ii < m ) && isdone )
                {
                    if ( ss(ii) == 1 )
                    {
                        ss("&",ii) = -1;
                        isdone = 0;
                    }

                    else
                    {
                        ss("&",ii) = +1;
                        ii++;
                    }
                }
            }

            res /= (1<<(m-1));
        }

        else if ( isAltDiff() == 104 )
        {
            Vector<int> ss(m);
            int isdone = 0;
            int cnt = 0;
            int z = 0;

            ss = z;

            while ( !isdone )
            {
                int noreps = 1;

                for ( ii = 0 ; ( ii < m ) && noreps ; ii++ )
                {
                    for ( jj = ii+1 ; ( jj < m ) && noreps ; jj++ )
                    {
                        if ( ss(ii) == ss(jj) )
                        {
                            noreps = 0;
                        }
                    }
                }

                if ( noreps )
                {
                    cnt++;
                    LLm(m,tempres,dummya,dummyb,x,xinfo,bias,i,NULL,xdim,xconsist,assumreal,resmode,mlid,xy,&ss,0,indstart,indend);
                    res += tempres;
                }

                isdone = 1;
                ii = 0;

                while ( ( ii < m ) && isdone )
                {
                    ss("&",ii)++;

                    if ( ss(ii) < m )
                    {
                        isdone = 0;
                    }

                    else
                    {
                        ss("&",ii) = z;
                        ii++;
                    }
                }
            }

            res /= cnt;
        }
    }

    else if ( ( resmode & 0x01 ) )
    {
        NiceAssert( !justcalcip );
        NiceAssert( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) );
        NiceAssert( !( isprod && !arexysimple(m,x) ) );

        Vector<const gentype *> xnormde(x.size());

        if ( m )
        {
            int ii;

            for ( ii = 0 ; ii < m ; ii++ )
            {
                retVector<const gentype *> tmpva;

                xnormde("&",i,tmpva) = &getmnorm(*(xinfo(ii)),*(x(ii)),m,xconsist,assumreal);
            }
        }

        Matrix<double> altxy;

        gentype tempres;

        Kmi(tempres,defaultgentype(),xinfo,xnormde,x,i,xdim,x.size(),0,resmode,mlid,fillXYMatrix(m,altxy,x,xinfo,xy,0,assumreal),s,indstart,indend,assumreal);

        res = (T) tempres;
    }

    else if ( isprod && !arexysimple(m,x) )
    {
        NiceAssert( !justcalcip );
        NiceAssert( !( resmode & 0x80 ) );

        // Kernel is of the form prod_i K(x_i,y_i)
        // x and y are not simple

        if ( xconsist && ( size() == 1 ) && isSimpleKernel() )
        {
            //NiceAssert( x.nearindsize() == y.nearindsize() );

            Vector<SparseVector<gentype> > xx(x.size());
            Vector<const SparseVector<gentype> *> xxx(x.size());

            Vector<vecInfo> xxinfo(x.size());
            Vector<const vecInfo *> xxxinfo(x.size());

            if ( m && (*(x(zeroint()))).nearindsize() )
            {
                int ii,jj;
                T tempres;

                for ( jj = 0 ; jj < m ; jj++ )
                {
                    xxx    ("&",jj) = &(xx    (jj));
                    xxxinfo("&",jj) = &(xxinfo(jj));
                }

                for ( ii = 0 ; ii < (*(x(zeroint()))).nearindsize() ; ii++ )
                {
                    for ( jj = 0 ; jj < m ; jj++ )
                    {
                        xx("&",jj)("&",zeroint()) = (*(x(jj))).direcref(ii);
                        getvecInfo(xxinfo("&",jj),xx(jj),NULL,xconsist,assumreal);
                    }

                    LLm(m,tempres,dummya,dummyb,xxx,xxxinfo,bias,i,NULL,1,1,resmode,assumreal,mlid,NULL,NULL,0,indstart,indend);

                    tempres /= (const T &) cWeight(0);

                    if ( !ii ) { res =  tempres; }
                    else       { res *= tempres; }
                }
            }

            else
            {
                res = 1;
            }
        }

        else
        {
            SparseVector<gentype> indres;

            combind(m,indres,x);

            Vector<SparseVector<gentype> > xx(x.size());
            Vector<const SparseVector<gentype> *> xxx(x.size());

            Vector<vecInfo> xxinfo(x.size());
            Vector<const vecInfo *> xxxinfo(x.size());

            if ( m && indres.size() )
            {
                int ii,jj;
                T tempres;

                for ( jj = 0 ; jj < m ; jj++ )
                {
                    xxx    ("&",jj) = &(xx    (jj));
                    xxxinfo("&",jj) = &(xxinfo(jj));
                }

                for ( ii = 0 ; ii < indres.nearindsize() ; ii++ )
                {
                    for ( jj = 0 ; jj < m ; jj++ )
                    {
                        xx("&",jj)("&",zeroint()) = (*(x(jj))).direcref(indres.ind(ii));
                        getvecInfo(xxinfo("&",jj),xx(jj),NULL,xconsist,assumreal);
                    }

                    LLm(m,tempres,dummya,dummyb,xxx,xxxinfo,bias,i,NULL,1,1,assumreal,resmode,mlid,NULL,NULL,0,indstart,indend);

                    tempres /= (const T &) cWeight(0);

                    if ( !ii ) { res =  tempres; }
                    else       { res *= tempres; }
                }
            }

            else
            {
                res = 1;
            }
        }

        res *= (const T &) cWeight(0);
    }

    else if ( ( isFastKernelSum() || isFastKernelChain() ) && ( resmode && 0x80 ) && !justcalcip )
    {
        res = 0;
    }

    else if ( isFastKernelSum() || isFastKernelChain() || ( isFastKernelXfer() && !resmode ) || justcalcip )
    {
        int locindstart = (isFastKernelXfer()?1:indstart); // NB: isFastKernelXfer() implies indstart == 0
        int locindend   = indend; //size()-1;

        int needxxprod = isNormalised(locindend);

        T xyprod; xyprod = 0.0;
        Vector<T> xxprod(x.size());
        T diffis; diffis = 0.0;

        xxprod = diffis; // diffis = 0.0

        if ( isFastKernelSum() || isFastKernelChain() || justcalcip )
        {
            if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
            }

            else if ( needsInner(0,m) || ( isFastKernelSum() && needsInner(-1,m) ) || justcalcip )
            {
                mProductDiverted(m,xyprod,x,xconsist,assumreal);

                if ( !justcalcip )
                {
                    xyprod += bias;
                }
            }

            if ( needxxprod && !justcalcip )
            {
                int ii;
                Vector<int> iii(x.size());

                for ( ii = 0 ; ii < m ; ii++ )
                {
                    iii = ii;

                    retVector<const SparseVector<gentype> *> tmpva;
                    retVector<const vecInfo *>               tmpvb;

                    mProductDiverted(m,xxprod("&",ii),x(iii,tmpva),xconsist,assumreal);

                    if ( !justcalcip )
                    {
                        xxprod("&",ii) += bias;
                    }
                }
            }

            if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else if ( ( ( needsDiff(0) || ( isFastKernelSum() && needsDiff() ) ) && ( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) ) ) && !justcalcip )
            {
                Vector<T> xnormrr(x.size());
                Vector<const T *> xnormde(x.size());

                if ( m )
                {
                    int ii;

                    for ( ii = 0 ; ii < m ; ii++ )
                    {
                        xnormrr("&",ii) = getmnorm(*(xinfo(ii)),*(x(ii)),m,xconsist,assumreal);
                        xnormde("&",ii) = &xnormrr(ii); //&getmnorm(*(xinfo(ii)),*(x(ii)),m,xconsist,assumreal);
                    }
                }

                Matrix<double> altxy;

                diffmnorm(m,diffis,xyprod,xnormde,fillXYMatrix(m,altxy,x,xinfo,xy,0,assumreal),s);
            }

            else if ( ( needsDiff(0) && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) && !justcalcip )
            {
                // At this point we need to calculate diffis using altdiffis method 2xx
                // xyprod is not used by kernel, but need to fill it in for use by rest of chain
                // we only need to cycle through diffis for relevant s vectors
                // We need the xy matrix to do this

                T tempres;
                int ii,jj;

                Matrix<double> altxy(m,m);
                const Matrix<double> *sxy = &fillXYMatrix(m,altxy,x,xinfo,xy,1,assumreal);
                const Matrix<double> &ssxy = *sxy;

                xyprod = 0.0;

                if ( isAltDiff() == 203 )
                {
                    T dummy;

                    Vector<int> ss(m);

                    int isdone = 0;

                    ss = 1;

                    while ( !isdone )
                    {
                        if ( sum(ss)%4 == 0 )
                        {
                            diffis = 0.0;

                            for ( ii = 0 ; ii < m ; ii++ )
                            {
                                for ( jj = 0 ; jj < m ; jj++ )
                                {
                                    diffis += ss(ii)*ss(jj)*ssxy(ii,jj);
                                }
                            }

                            KKpro(tempres,xyprod,diffis,&(i("&",zeroint())),locindstart,locindstart,xdim,m,dummy,&dummy);
                            xyprod += tempres;
                        }

                        isdone = 1;
                        ii = 0;

                        while ( ( ii < m ) && isdone )
                        {
                            if ( ss(ii) == 1 )
                            {
                                ss("&",ii) = -1;
                                isdone = 0;
                            }

                            else
                            {
                                ss("&",ii) = +1;
                                ii++;
                            }
                        }
                    }

                    xyprod /= (1<<(m-1));
                }

                else if ( isAltDiff() == 204 )
                {
                    Vector<int> ss(m);

                    int isdone = 0;
                    int cnt = 0;
                    int z = 0;

                    ss = z;

                    while ( !isdone )
                    {
                        T dummy;

                        int noreps = 1;

                        for ( ii = 0 ; ( ii < m ) && noreps ; ii++ )
                        {
                            for ( jj = ii+1 ; ( jj < m ) && noreps ; jj++ )
                            {
                                if ( ss(ii) == ss(jj) )
                                {
                                    noreps = 0;
                                }
                            }
                        }

                        if ( noreps )
                        {
                            diffis = 0.0;

                            for ( ii = 0 ; ii < m ; ii += 2 )
                            {
                                diffis +=  ssxy(ss(ii  ),ss(ii  )) - ssxy(ss(ii  ),ss(ii+1));
                                diffis += -ssxy(ss(ii+1),ss(ii  )) + ssxy(ss(ii+1),ss(ii+1));
                            }

                            KKpro(tempres,xyprod,diffis,&(i("&",zeroint())),locindstart,locindstart,xdim,m,dummy,&dummy);
                            xyprod += tempres;

                            cnt++;
                        }

                        isdone = 1;
                        ii = 0;

                        while ( ( ii < m ) && isdone )
                        {
                            ss("&",ii)++;

                            if ( ss(ii) < m )
                            {
                                isdone = 0;
                            }

                            else
                            {
                                ss("&",ii) = z;
                                ii++;
                            }
                        }
                    }

                    xyprod /= cnt;
                }

                locindstart++;
            }
        }

        else
        {
            // ( isFastKernelXfer() && !resmode )
            NiceAssert( !( resmode & 0x80 ) );

            int dummyind = 0;

            if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
            }

            else
            {
                kernel8xx(0,xyprod,dummyind,cType(zeroint()),x,xinfo,i,xdim,m,0,0,mlid);
            }

            if ( needxxprod && !justcalcip )
            {
                int ii;
                Vector<int> iii(x.size());

                for ( ii = 0 ; ii < m ; ii++ )
                {
                    iii = ii;

                    retVector<const SparseVector<gentype> *> tmpva;
                    retVector<const vecInfo *>               tmpvb;
                    retVector<int>                           tmpvc;

                    kernel8xx(0,xxprod("&",ii),dummyind,cType(zeroint()),x("&",iii,tmpva),xinfo("&",iii,tmpvb),i("&",iii,tmpvc),xdim,m,0,0,mlid);
                }
            }

            if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else if ( ( size() >= 2 ) && needsDiff(1) && ( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) ) && !justcalcip )
            {
                Matrix<double> altxy;

                Vector<T> xnormdex(x.size());
                Vector<const T *> xnormde(x.size());

                Vector<const SparseVector<gentype> *> xx(x.size());
                Vector<const vecInfo *> xxinfo(xinfo.size());
                Vector<int> iii(i.size());

                if ( needsMatDiff() )
                {
                    altxy.resize(m,m);
                }

                if ( m )
                {
                    int ii,jj;

                    for ( ii = 0 ; ii < m ; ii++ )
                    {
                        xx = x(ii);
                        xxinfo = xinfo(ii);
                        iii = i(ii);

                        if ( isAltDiff() == 0 )
                        {
                            kernel8xx(0,xnormdex("&",ii),dummyind,cType(zeroint()),
                                          xx,xxinfo,iii,
                                          xdim,m,0,0,mlid);
                        }

                        else
                        {
                            kernel8xx(0,xnormdex("&",ii),dummyind,cType(zeroint()),*x(ii),*x(ii),*xinfo(ii),*xinfo(ii),i(ii),i(ii),xdim,0,0,mlid);

                            if ( needsMatDiff() )
                            {
                                altxy("&",ii,ii) = xnormdex("&",ii);

                                if ( ii )
                                {
                                    for ( jj = 0 ; jj < ii ; jj++ )
                                    {
                                        if ( ii != jj )
                                        {
                                            kernel8xx(0,altxy("&",ii,jj),dummyind,cType(zeroint()),*x(ii),*x(jj),*xinfo(ii),*xinfo(jj),i(ii),i(jj),xdim,0,0,mlid);
                                                altxy("&",jj,ii) = altxy("&",ii,jj);
                                                setconj(altxy("&",jj,ii));
                                        }
                                    }
                                }
                            }
                        }

                        xnormde("&",ii) = &(xnormdex(ii));
                    }
                }

                diffmnorm(m,diffis,xyprod,xnormde,altxy,s);

                diffis *= (const T &) cWeight(0);
            }

            else if ( needsDiff(0) && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) && !justcalcip )
            {
                // At this point we need to calculate diffis using altdiffis method 2xx
                // xyprod is not used by kernel
                // we only need to cycle through diffis for relevant s vectors
                // We need the xy matrix to do this

                T tempres; tempres = 0.0;

                Matrix<double> altxy(m,m);
                Matrix<double> &ssxy = altxy;

                Vector<T> xnormdex(x.size());
                Vector<const T *> xnormde(x.size());

                Vector<const SparseVector<gentype> *> xx(x.size());
                Vector<const vecInfo *> xxinfo(xinfo.size());
                Vector<int> iii(i.size());

                if ( m )
                {
                    int ii,jj;
                    int dummyind = 0;

                    for ( ii = 0 ; ii < m ; ii++ )
                    {
                        xx = x(ii);
                        xxinfo = xinfo(ii);
                        iii = i(ii);

                        kernel8xx(0,xnormdex("&",ii),dummyind,cType(zeroint()),*x(ii),*x(ii),*xinfo(ii),*xinfo(ii),i(ii),i(ii),xdim,0,0,mlid);

                        if ( needsMatDiff() )
                        {
                            altxy("&",ii,ii) = xnormdex("&",ii);

                            if ( ii )
                            {
                                for ( jj = 0 ; jj < ii ; jj++ )
                                {
                                    if ( ii != jj )
                                    {
                                        kernel8xx(0,altxy("&",ii,jj),dummyind,cType(zeroint()),*x(ii),*x(jj),*xinfo(ii),*xinfo(jj),i(ii),i(jj),xdim,0,0,mlid);
                                            altxy("&",jj,ii) = altxy("&",ii,jj);
                                            setconj(altxy("&",jj,ii));
                                    }
                                }
                            }
                        }
                    }
                }

                xyprod = 0.0;

                if ( isAltDiff() == 203 )
                {
                    T dummy;

                    Vector<int> ss(m);
                    int isdone = 0;
                    int ii,jj;

                    ss = 1;

                    while ( !isdone )
                    {
                        if ( sum(ss)%4 == 0 )
                        {
                            diffis = 0.0;

                            for ( ii = 0 ; ii < m ; ii++ )
                            {
                                for ( jj = 0 ; jj < m ; jj++ )
                                {
                                    diffis += ss(ii)*ss(jj)*ssxy(ii,jj);
                                }
                            }

                            KKpro(tempres,xyprod,diffis,&(i("&",zeroint())),locindstart,locindstart,xdim,m,dummy,&dummy);
                            xyprod += tempres;
                        }

                        isdone = 1;
                        ii = 0;

                        while ( ( ii < m ) && isdone )
                        {
                            if ( ss(ii) == 1 )
                            {
                                ss("&",ii) = -1;
                                isdone = 0;
                            }

                            else
                            {
                                ss("&",ii) = +1;
                                ii++;
                            }
                        }
                    }

                    xyprod /= (1<<(m-1));
                }

                else if ( isAltDiff() == 204 )
                {
                    Vector<int> ss(m);
                    int isdone = 0;
                    int cnt = 0;
                    int z = 0;
                    int ii,jj;

                    ss = z;

                    while ( !isdone )
                    {
                        T dummy;

                        int noreps = 1;

                        for ( ii = 0 ; ( ii < m ) && noreps ; ii++ )
                        {
                            for ( jj = ii+1 ; ( jj < m ) && noreps ; jj++ )
                            {
                                if ( ss(ii) == ss(jj) )
                                {
                                    noreps = 0;
                                }
                            }
                        }

                        if ( noreps )
                        {
                            diffis = 0.0;

                            for ( ii = 0 ; ii < m ; ii += 2 )
                            {
                                diffis +=  ssxy(ss(ii  ),ss(ii  )) - ssxy(ss(ii  ),ss(ii+1));
                                diffis += -ssxy(ss(ii+1),ss(ii  )) + ssxy(ss(ii+1),ss(ii+1));
                            }

                            KKpro(tempres,xyprod,diffis,&(i("&",zeroint())),locindstart,locindstart,xdim,m,dummy,&dummy);
                            xyprod += tempres;

                            cnt++;
                        }

                        isdone = 1;
                        ii = 0;

                        while ( ( ii < m ) && isdone )
                        {
                            ss("&",ii)++;

                            if ( ss(ii) < m )
                            {
                                isdone = 0;
                            }

                            else
                            {
                                ss("&",ii) = z;
                                ii++;
                            }
                        }
                    }

                    xyprod /= cnt;
                }

                locindstart++;
            }

            diffis *= (const T &) cWeight(0);
        }

        if ( justcalcip )
        {
            res = xyprod;
        }

        else
        {
            if ( isNormalised(locindend) )
            {
                T dummy;

                KKpro(res,xyprod,diffis,&(i("&",zeroint())),locindstart,locindend,xdim,m,dummy,&dummy);

                T zerodiff; zerodiff = 0.0;
                int ii;
                Vector<int> iii(x.size());
                T xxres;
                T xxxres;
                T oneonm; oneonm = 1.0/m;

                xxxres = 1.0;

                Vector<int> iw(m);

                for ( ii = 0 ; ii < m ; ii++ )
                {
                    iw = i(ii);

                    iii = ii;

                    KKpro(xxres,xxprod(ii),zerodiff,&(iw("&",zeroint())),locindstart,locindend,xdim,m,dummy,&dummy);

                    if ( !ii )
                    {
                        xxxres = xxres;
                    }

                    else
                    {
                        xxxres *= xxres;
                    }
                }

                safedivby(res,pow(xxxres,oneonm)); //res /= pow(xxres,oneonm);
            }

            else
            {
                T dummy;

                logresvalid = KKpro(res,xyprod,diffis,&(i("&",zeroint())),locindstart,locindend,xdim,m,logres,&dummy);
            }
        }
    }

    else
    {
        NiceAssert( ismagterm == zeroint() );
        NiceAssert( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) );

        gentype xyprod(0.0);

        if ( pxyprod && pxyprod[0] )
        {
            xyprod = *pxyprod[0];
        }

        else if ( needsInner(-1,m) )
        {
            // This may be used by some kernels and not others, so calculate anyhow

            mProductDiverted(m,xyprod,x,xconsist,assumreal);

            xyprod += bias;
        }

        Vector<const gentype *> xnormde(x.size());

        if ( m )
        {
            int ii;

            for ( ii = 0 ; ii < m ; ii++ )
            {
                xnormde("&",ii) = &getmnorm(*(xinfo(ii)),*(x(ii)),m,xconsist,assumreal);
            }
        }

        Matrix<double> altxy;

        gentype tempres;

        Kmi(tempres,xyprod,xinfo,xnormde,x,i,xdim,m,0,0,mlid,fillXYMatrix(m,altxy,x,xinfo,xy,0,assumreal),s,indstart,indend,assumreal);

        res = (T) tempres;
    }

    return res;

badout:
    // Design decision: in ml_base.cc, if d = 0 for one of the vectors
    // referenced here then this element will never be used.  Moreover there
    // are cases (eg isAltDiff set >1 with back-referenced data) where the
    // element is not properly defined but will never be used, so what we 
    // need to do is set it 0.  However having a "d = 0" catch will fail when
    // d starts non-zero, is set zero, then set non-zero, as will happen for
    // example when calculating LOO, n-fold error etc.  In such cases you need
    // to call a reset on that row/column, but you can't do that because (a) the
    // reset often calls setd (hence infinite recursion) or (b) there is an 
    // implicit assumption that Gp is independent of d (eg in semicopy functions
    // that retain the caches for speed in LOO, n-fold calculation).  Hence I've 
    // made the decision to return 0 here to avoid a whole stack of potential
    // coding complications at the price of possible silent failure if you set
    // somthing incorrectly.

    errstream() << "!!!modd!!!";

    res = 0.0;

    return res;
}













template <class T>
void MercerKernel::xdKK2(T &xygrad, T &xnormgrad, int &minmaxind, 
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                         const vecInfo &xainfo, const vecInfo &xbinfo, 
                         const T &bias, 
                         const gentype **pxyprod, 
                         int ia, int ib, 
                         int xdim, int xconsist, int assumreal, int mlid, 
                         const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset) const
{
    // isfullnorm should happen here, but doesn't
    NiceAssert( !isfullnorm );

    dKK2(xygrad,xnormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    return;
}

template <class T>
void MercerKernel::xd2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, 
                          const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                          const vecInfo &xainfo, const vecInfo &xbinfo, 
                          const T &bias, 
                          const gentype **pxyprod, 
                          int ia, int ib, 
                          int xdim, int xconsist, int assumreal, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset) const
{
    // isfullnorm should happen here, but doesn't
    NiceAssert( !isfullnorm );

    d2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    return;
}

template <class T>
void MercerKernel::xdnKK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, 
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                             const vecInfo &xainfo, const vecInfo &xbinfo, 
                             const T &bias, 
                             const gentype **pxyprod, 
                             int ia, int ib, 
                             int xdim, int xconsist, int assumreal, int mlid, 
                             const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int iaset, int ibset) const
{
    // isfullnorm should happen here, but doesn't
    NiceAssert( !isfullnorm );

    dnKK2del(sc,n,minmaxind,q,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset);

    return;
}







//phantomx
template <class T>
void MercerKernel::dKK2(T &xygrad, T &xnormgrad, int &minmaxind,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                    const vecInfo &xainfo, const vecInfo &xbinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib,
                    int xdim, int xconsist, int assumreal, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, int deepDeriv, 
                    int iaset, int ibset,
                    int skipbias,
                    int skipxa, int skipxb) const
{
    if ( !skipbias && isiteqn(bias) )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qb;
        int maxq = numSamples();

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qb = 0 ; qb < maxq ; qb++ )
        {
            gentype gbias(bias);

            if ( !subSample(subval,gbias) && !qb )
            {
                goto postbias;
            }

            gentype gxygrad,gxnormgrad;

            dKK2(gxygrad,gxnormgrad,minmaxind,xa,xb,xainfo,xbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,mlid,NULL,NULL,NULL,deepDeriv,iaset,ibset,1,skipxa,skipxb);

            if ( !qb ) { xygrad =  (T) gxygrad; xnormgrad =  (T) gxnormgrad; }
            else       { xygrad += (T) gxygrad; xnormgrad += (T) gxnormgrad; }
        }

        xygrad    /= maxq;
        xnormgrad /= maxq;

        return;
    }

postbias:

    if ( !skipxa && xainfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        T res; res = 0.0;
        int logresvalid = 1;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxa(xa);
            vecInfo xxainfo;

            if ( !subSample(subval,xxa,xxainfo) && !qxa )
            {
                goto postxa;
            }

            gentype gres,glogres;

            KK2(gres,glogres,logresvalid,xxa,xb,xxainfo,xbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,0,mlid,NULL,NULL,NULL,0,0,size()-1,iaset,ibset);

            gentype gxygrad,gxnormgrad;

            dKK2(gxygrad,gxnormgrad,minmaxind,xxa,xb,xxainfo,xbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,mlid,NULL,NULL,NULL,deepDeriv,iaset,ibset,skipbias,1,skipxb);

            if ( !qxa )                    { res =  (T) gres; xygrad =  (T) gxygrad; xnormgrad =  (T) gxnormgrad; }
            else if ( !iaset )             { res += (T) gres; xygrad += (T) gxygrad; xnormgrad += (T) gxnormgrad; }
            else if ( (T) gres > (T) res ) { res =  (T) gres; xygrad =  (T) gxygrad; xnormgrad =  (T) gxnormgrad; }
        }

        if ( !iaset )
        {
            xygrad    /= maxq;
            xnormgrad /= maxq;
        }

        return;
    }

postxa:

    if ( !skipxb && xbinfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        T res; res = 0.0;
        int logresvalid = 1;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxb(xb);
            vecInfo xxbinfo;

            if ( !subSample(subval,xxb,xxbinfo) && !qxa )
            {
                goto postxb;
            }

            gentype gres,glogres;

            KK2(gres,glogres,logresvalid,xa,xxb,xainfo,xxbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,0,mlid,NULL,NULL,NULL,0,0,size()-1,iaset,ibset);

            gentype gxygrad,gxnormgrad;

            dKK2(gxygrad,gxnormgrad,minmaxind,xa,xxb,xainfo,xxbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,mlid,NULL,NULL,NULL,deepDeriv,iaset,ibset,skipbias,skipxa,1);

            if ( !qxa )                    { res =  (T) gres; xygrad =  (T) gxygrad; xnormgrad =  (T) gxnormgrad; }
            else if ( !ibset )             { res += (T) gres; xygrad += (T) gxygrad; xnormgrad += (T) gxnormgrad; }
            else if ( (T) gres > (T) res ) { res =  (T) gres; xygrad =  (T) gxygrad; xnormgrad =  (T) gxnormgrad; }
        }

        if ( !ibset )
        {
            xygrad    /= maxq;
            xnormgrad /= maxq;
        }

        return;
    }

postxb:

    dLL2(xygrad,xnormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}

//phantomx
template <class T>
void MercerKernel::d2KK2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, 
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                    const vecInfo &xainfo, const vecInfo &xbinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int ia, int ib,
                    int xdim, int xconsist, int assumreal, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, int deepDeriv, 
                    int iaset, int ibset,
                    int skipbias,
                    int skipxa, int skipxb) const
{
    if ( !skipbias && isiteqn(bias) )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qb;
        int maxq = numSamples();

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qb = 0 ; qb < maxq ; qb++ )
        {
            gentype gbias(bias);

            if ( !subSample(subval,gbias) && !qb )
            {
                goto postbias;
            }

            gentype tempxygrad;
            gentype tempxnormgrad;
            gentype tempxyxygrad;
            gentype tempxyxnormgrad;
            gentype tempxyynormgrad;
            gentype tempxnormxnormgrad;
            gentype tempxnormynormgrad;
            gentype tempynormynormgrad;

            d2KK2(tempxygrad,tempxnormgrad,tempxyxygrad,tempxyxnormgrad,tempxyynormgrad,tempxnormxnormgrad,tempxnormynormgrad,tempynormynormgrad,minmaxind,xa,xb,xainfo,xbinfo,gbias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,NULL,NULL,NULL,deepDeriv,iaset,ibset,1,skipxa,skipxb);

            if ( !qb )
            {
                xygrad         = (T) tempxygrad;
                xnormgrad      = (T) tempxnormgrad;
                xyxygrad       = (T) tempxyxygrad;
                xyxnormgrad    = (T) tempxyxnormgrad;
                xyynormgrad    = (T) tempxyynormgrad;
                xnormxnormgrad = (T) tempxnormxnormgrad;
                xnormynormgrad = (T) tempxnormynormgrad;
                ynormynormgrad = (T) tempynormynormgrad;
            }

            else
            {
                xygrad         += (T) tempxygrad;
                xnormgrad      += (T) tempxnormgrad;
                xyxygrad       += (T) tempxyxygrad;
                xyxnormgrad    += (T) tempxyxnormgrad;
                xyynormgrad    += (T) tempxyynormgrad;
                xnormxnormgrad += (T) tempxnormxnormgrad;
                xnormynormgrad += (T) tempxnormynormgrad;
                ynormynormgrad += (T) tempynormynormgrad;
            }
        }

        xygrad         /= maxq;
        xnormgrad      /= maxq;
        xyxygrad       /= maxq;
        xyxnormgrad    /= maxq;
        xyynormgrad    /= maxq;
        xnormxnormgrad /= maxq;
        xnormynormgrad /= maxq;
        ynormynormgrad /= maxq;

        return;
    }

postbias:

    if ( !skipxa && xainfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        T res; res = 0.0;
        int logresvalid = 1;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxa(xa);
            vecInfo xxainfo;

            if ( !subSample(subval,xxa,xxainfo) && !qxa )
            {
                goto postxa;
            }

            gentype gres,glogres;

            KK2(gres,glogres,logresvalid,xxa,xb,xxainfo,xbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,0,mlid,NULL,NULL,NULL,0,0,size()-1,iaset,ibset);

            gentype tempxygrad;
            gentype tempxnormgrad;
            gentype tempxyxygrad;
            gentype tempxyxnormgrad;
            gentype tempxyynormgrad;
            gentype tempxnormxnormgrad;
            gentype tempxnormynormgrad;
            gentype tempynormynormgrad;

            d2KK2(tempxygrad,tempxnormgrad,tempxyxygrad,tempxyxnormgrad,tempxyynormgrad,tempxnormxnormgrad,tempxnormynormgrad,tempynormynormgrad,minmaxind,xxa,xb,xxainfo,xbinfo,gbias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,NULL,NULL,NULL,deepDeriv,iaset,ibset,skipbias,1,skipxb);

            if ( !qxa )
            {
                res            = (T) gres;
                xygrad         = (T) tempxygrad;
                xnormgrad      = (T) tempxnormgrad;
                xyxygrad       = (T) tempxyxygrad;
                xyxnormgrad    = (T) tempxyxnormgrad;
                xyynormgrad    = (T) tempxyynormgrad;
                xnormxnormgrad = (T) tempxnormxnormgrad;
                xnormynormgrad = (T) tempxnormynormgrad;
                ynormynormgrad = (T) tempynormynormgrad;
            }

            else if ( !iaset )
            {
                res            += (T) gres;
                xygrad         += (T) tempxygrad;
                xnormgrad      += (T) tempxnormgrad;
                xyxygrad       += (T) tempxyxygrad;
                xyxnormgrad    += (T) tempxyxnormgrad;
                xyynormgrad    += (T) tempxyynormgrad;
                xnormxnormgrad += (T) tempxnormxnormgrad;
                xnormynormgrad += (T) tempxnormynormgrad;
                ynormynormgrad += (T) tempynormynormgrad;
            }

            else if ( (T) gres > (T) res )
            {
                res            = (T) gres;
                xygrad         = (T) tempxygrad;
                xnormgrad      = (T) tempxnormgrad;
                xyxygrad       = (T) tempxyxygrad;
                xyxnormgrad    = (T) tempxyxnormgrad;
                xyynormgrad    = (T) tempxyynormgrad;
                xnormxnormgrad = (T) tempxnormxnormgrad;
                xnormynormgrad = (T) tempxnormynormgrad;
                ynormynormgrad = (T) tempynormynormgrad;
            }
        }

        if ( !iaset )
        {
            xygrad         /= maxq;
            xnormgrad      /= maxq;
            xyxygrad       /= maxq;
            xyxnormgrad    /= maxq;
            xyynormgrad    /= maxq;
            xnormxnormgrad /= maxq;
            xnormynormgrad /= maxq;
            ynormynormgrad /= maxq;
        }

        return;
    }

postxa:

    if ( !skipxb && xbinfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        T res; res = 0.0;
        int logresvalid = 1;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxb(xb);
            vecInfo xxbinfo;

            if ( !subSample(subval,xxb,xxbinfo) && !qxa )
            {
                goto postxb;
            }

            gentype gres,glogres;

            KK2(gres,glogres,logresvalid,xa,xxb,xainfo,xxbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,0,mlid,NULL,NULL,NULL,0,0,size()-1,iaset,ibset);

            gentype tempxygrad;
            gentype tempxnormgrad;
            gentype tempxyxygrad;
            gentype tempxyxnormgrad;
            gentype tempxyynormgrad;
            gentype tempxnormxnormgrad;
            gentype tempxnormynormgrad;
            gentype tempynormynormgrad;

            d2KK2(tempxygrad,tempxnormgrad,tempxyxygrad,tempxyxnormgrad,tempxyynormgrad,tempxnormxnormgrad,tempxnormynormgrad,tempynormynormgrad,minmaxind,xa,xxb,xainfo,xxbinfo,gbias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,NULL,NULL,NULL,deepDeriv,iaset,ibset,skipbias,skipxa,1);

            if ( !qxa )
            {
                res            = (T) gres;
                xygrad         = (T) tempxygrad;
                xnormgrad      = (T) tempxnormgrad;
                xyxygrad       = (T) tempxyxygrad;
                xyxnormgrad    = (T) tempxyxnormgrad;
                xyynormgrad    = (T) tempxyynormgrad;
                xnormxnormgrad = (T) tempxnormxnormgrad;
                xnormynormgrad = (T) tempxnormynormgrad;
                ynormynormgrad = (T) tempynormynormgrad;
            }

            else if ( !ibset )
            {
                res            += (T) gres;
                xygrad         += (T) tempxygrad;
                xnormgrad      += (T) tempxnormgrad;
                xyxygrad       += (T) tempxyxygrad;
                xyxnormgrad    += (T) tempxyxnormgrad;
                xyynormgrad    += (T) tempxyynormgrad;
                xnormxnormgrad += (T) tempxnormxnormgrad;
                xnormynormgrad += (T) tempxnormynormgrad;
                ynormynormgrad += (T) tempynormynormgrad;
            }

            else if ( (T) gres > (T) res )
            {
                res            = (T) gres;
                xygrad         = (T) tempxygrad;
                xnormgrad      = (T) tempxnormgrad;
                xyxygrad       = (T) tempxyxygrad;
                xyxnormgrad    = (T) tempxyxnormgrad;
                xyynormgrad    = (T) tempxyynormgrad;
                xnormxnormgrad = (T) tempxnormxnormgrad;
                xnormynormgrad = (T) tempxnormynormgrad;
                ynormynormgrad = (T) tempynormynormgrad;
            }
        }

        if ( !ibset )
        {
            xygrad         /= maxq;
            xnormgrad      /= maxq;
            xyxygrad       /= maxq;
            xyxnormgrad    /= maxq;
            xyynormgrad    /= maxq;
            xnormxnormgrad /= maxq;
            xnormynormgrad /= maxq;
            ynormynormgrad /= maxq;
        }

        return;
    }

postxb:

    d2LL2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}

template <class T>
void MercerKernel::dnKK2del(Vector<T> &sc, Vector<Vector<int> > &nn, int &minmaxind, 
                           const Vector<int> &qq, 
                           const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                           const vecInfo &xainfo, const vecInfo &xbinfo, 
                           const T &bias, const gentype **pxyprod, 
                           int ia, int ib, 
                           int xdim, int xconsist, int assumreal, int mlid, 
                           const double *xy00, const double *xy10, const double *xy11, int deepDeriv,
                           int iaset, int ibset,
                           int skipbias,
                           int skipxa, int skipxb) const
{
    int t;

    if ( !skipbias && isiteqn(bias) )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qb;
        int maxq = numSamples();

        SparseVector<SparseVector<gentype> > subval;

        // Take maxq samples from output distribution

        for ( qb = 0 ; qb < maxq ; qb++ )
        {
            gentype gbias(bias);

            if ( !subSample(subval,gbias) && !qb )
            {
                goto postbias;
            }

            Vector<gentype> tempsc(sc.size());

            dnKK2del(tempsc,nn,minmaxind,qq,xa,xb,xainfo,xbinfo,gbias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset,1,skipxa,skipxb);

            if ( !qb )
            {
                sc.resize(tempsc.size());

                for ( t = 0 ; t < sc.size() ; t++ )
                {
                    sc("&",t) = (T) tempsc(t);
                }
            }

            else
            {
                for ( t = 0 ; t < sc.size() ; t++ )
                {
                    sc("&",t) += (T) tempsc(t);
                }
            }
        }

        for ( t = 0 ; t < sc.size() ; t++ )
        {
            sc("&",t) /= maxq;
        }

        return;
    }

postbias:

    if ( !skipxa && xainfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        T res; res = 0.0;
        int logresvalid = 1;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxa(xa);
            vecInfo xxainfo;

            if ( !subSample(subval,xxa,xxainfo) && !qxa )
            {
                goto postxa;
            }

            gentype gres,glogres;

            KK2(gres,glogres,logresvalid,xxa,xb,xxainfo,xbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,0,mlid,NULL,NULL,NULL,0,0,size()-1,iaset,ibset);

            Vector<gentype> tempsc(sc.size());

            dnKK2del(tempsc,nn,minmaxind,qq,xxa,xb,xxainfo,xbinfo,gbias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset,skipbias,1,skipxb);

            if ( !qxa )
            {
                res = (T) gres;

                sc.resize(tempsc.size());

                for ( t = 0 ; t < sc.size() ; t++ )
                {
                    sc("&",t) = (T) tempsc(t);
                }
            }

            else if ( !iaset )
            {
                res += (T) gres;

                for ( t = 0 ; t < sc.size() ; t++ )
                {
                    sc("&",t) += (T) tempsc(t);
                }
            }

            else if ( (T) gres > (T) res )
            {
                res = (T) gres;

                for ( t = 0 ; t < sc.size() ; t++ )
                {
                    sc("&",t) = (T) tempsc(t);
                }
            }
        }

        if ( !iaset )
        {
            for ( t = 0 ; t < sc.size() ; t++ )
            {
                sc("&",t) /= maxq;
            }
        }

        return;
    }

postxa:

    if ( !skipxb && xbinfo.xiseqn() )
    {
        // We are dealing with distributions, so need to delay finalisation 
        // of random parts of the function and then average *outside* the loop
        //
        // See Muandet et al, Learning from Distributions via Support Measure Machines

        int qxa;
        int maxq = numSamples();

        gentype gbias(bias);

        SparseVector<SparseVector<gentype> > subval;

        T res; res = 0.0;
        int logresvalid = 1;

        // Take maxq samples from output distribution

        for ( qxa = 0 ; qxa < maxq ; qxa++ )
        {
            SparseVector<gentype> xxb(xb);
            vecInfo xxbinfo;

            if ( !subSample(subval,xxb,xxbinfo) && !qxa )
            {
                goto postxb;
            }

            gentype gres,glogres;

            KK2(gres,glogres,logresvalid,xa,xxb,xainfo,xxbinfo,gbias,NULL,ia,ib,xdim,xconsist,assumreal,0,mlid,NULL,NULL,NULL,0,0,size()-1,iaset,ibset);

            Vector<gentype> tempsc(sc.size());

            dnKK2del(tempsc,nn,minmaxind,qq,xa,xxb,xainfo,xxbinfo,gbias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv,iaset,ibset,skipbias,skipxa,1);

            if ( !qxa )
            {
                res = (T) gres;

                sc.resize(tempsc.size());

                for ( t = 0 ; t < sc.size() ; t++ )
                {
                    sc("&",t) = (T) tempsc(t);
                }
            }

            else if ( !ibset )
            {
                res += (T) gres;
 
                for ( t = 0 ; t < sc.size() ; t++ )
                {
                    sc("&",t) += (T) tempsc(t);
                }
            }

            else if ( (T) gres > (T) res )
            {
                res = (T) gres;

                for ( t = 0 ; t < sc.size() ; t++ )
                {
                    sc("&",t) = (T) tempsc(t);
                }
            }
        }

        if ( !ibset )
        {
            for ( t = 0 ; t < sc.size() ; t++ )
            {
                sc("&",t) /= maxq;
            }
        }

        return;
    }

postxb:

    dnLL2del(sc,nn,minmaxind,qq,xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);

    return;
}
















//phantomx
template <class T>
void MercerKernel::dLL2(T &xygrad, T &xnormgrad, int &minmaxind,
                    const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                    const vecInfo &xinfo, const vecInfo &yinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int i, int j,
                    int xdim, int xconsist, int assumreal, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const
{
    T res;

    NiceAssert( !( isFastKernelSum() && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) );
    NiceAssert( !( isprod && !arexysimple(x,y) ) );

    minmaxind = -1;

    if ( isFastKernelSum() || isFastKernelChain() )
    {
        int needxxprod = isNormalised() || needsNorm();

        T xyprod; xyprod = 0.0;
        T yxprod; yxprod = 0.0;

        T xxprod; xxprod = 0.0;
        T yyprod; yyprod = 0.0;

        T diffis; diffis = 0.0;

        if ( xy10 )
        {
            xyprod = (*xy10);
            yxprod = (*xy10);

            xyprod += bias;
            yxprod += bias;
        }

        else if ( pxyprod && pxyprod[0] )
        {
            xyprod = *pxyprod[0];
            yxprod = xyprod;
        }

        else if ( needsInner(0,2) || ( isFastKernelSum() && needsInner(-1,2) ) )
        {
            twoProductDiverted(xyprod,x,y,xconsist,assumreal);
            twoProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

            xyprod += bias;
            yxprod += bias;
        }

        xyprod += yxprod;
        xyprod *= 0.5;

        if ( needxxprod )
        {
            if ( xy00 && xy11 )
            {
                xxprod = (*xy00);
                yyprod = (*xy11);

                xxprod += bias;
                yyprod += bias;
            }

            else
            {
                twoProductDiverted(xxprod,x,x,xconsist,assumreal);
                twoProductDiverted(yyprod,y,y,xconsist,assumreal);

                xxprod += bias;
                yyprod += bias;
            }
        }

        if ( ( needsDiff(0) || ( isFastKernelSum() && needsDiff() ) ) )
        {
            if ( xy00 && xy11 )
            {
                diff2norm(diffis,(double) xyprod,(*xy00),(*xy11));
            }

            else if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else
            {
                // Calculate ||x-y||^2 only as required

                if ( assumreal )
                {
                    diff2norm(diffis,(double) xyprod,(double) getmnorm(xinfo,x,2,xconsist,assumreal),(double) getmnorm(yinfo,y,2,xconsist,assumreal));
                }

                else
                {
                    diff2norm(diffis,(xyprod+yxprod)/2.0,getmnorm(xinfo,x,2,xconsist,assumreal),getmnorm(yinfo,y,2,xconsist,assumreal));
                }
            }
        }

        dKKpro(xygrad,xnormgrad,res,xyprod,diffis,i,j,0,size()-1,xdim,2,xxprod,yyprod);
    }

    else if ( isFastKernelXfer() )
    {
        NiceAssert( ismagterm == zeroint() );

        T xyprod; xyprod = 0.0;
        T yxprod; yxprod = 0.0;
        T diffis; diffis = 0.0;

        if ( ( !( size() >= 2 ) || !needsDiff(1) ) && deepDeriv )
        {
            dkernel8xx(0,xygrad,xnormgrad,xyprod,minmaxind,cType(zeroint()),x,y,xinfo,yinfo,i,j,xdim,0,48,mlid);

            xyprod *= (const T &) cWeight(0);

            xygrad    *= (const T &) cWeight(0);
            xnormgrad *= (const T &) cWeight(0);

            T dxyprod; dxyprod = 0.0;
            T dxnorm;  dxnorm  = 0.0;

            T dummy;

            dKKpro(dxyprod,dxnorm,res,xyprod,diffis,i,j,1,size()-1,xdim,2,dummy,dummy);

            xygrad    *= dxyprod;
            xnormgrad *= dxyprod;
        }

        else if ( ( !( size() >= 2 ) || !needsDiff(1) ) && !deepDeriv )
        {
            int dummyind = 0;

            if ( xy10 )
            {
                xyprod = (*xy10);
                yxprod = (*xy10);

                xyprod += bias;
                yxprod += bias;
            }

            else if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
                yxprod = xyprod;
            }

            else
            {
                kernel8xx(0,xyprod,dummyind,cType(zeroint()),x,y,xinfo,yinfo,i,j,xdim,0,0,mlid);

                xyprod += bias;
                yxprod  = xyprod;
            }

            xyprod += yxprod;
            xyprod *= 0.5;

            xyprod *= (const T &) cWeight(0);

            T dummy;

            dKKpro(xygrad,xnormgrad,res,xyprod,diffis,i,j,1,size()-1,xdim,2,dummy,dummy);
        }

        else if ( ( size() >= 2 ) && needsDiff(1) && deepDeriv )
        {
            T sxygrad;    sxygrad    = 0.0;
            T sxnormgrad; sxnormgrad = 0.0;

            dkernel8xx(0,sxygrad,sxnormgrad,xyprod,minmaxind,cType(zeroint()),x,y,xinfo,yinfo,i,j,xdim,0,48,mlid);

            xyprod *= (const T &) cWeight(0);

            xygrad    *= (const T &) cWeight(0);
            xnormgrad *= (const T &) cWeight(0);

            T xnorm;
            T ynorm;

            T dxxgrad; dxxgrad = 0.0;
            T dyygrad; dyygrad = 0.0;

            T dxnormgrad; dxnormgrad = 0.0;
            T dynormgrad; dynormgrad = 0.0;

            dkernel8xx(0,dxxgrad,dxnormgrad,xnorm,minmaxind,cType(zeroint()),x,x,xinfo,xinfo,i,i,xdim,0,48,mlid);
            dkernel8xx(0,dyygrad,dynormgrad,ynorm,minmaxind,cType(zeroint()),y,y,yinfo,yinfo,j,j,xdim,0,48,mlid);

            diff2norm(diffis,xyprod,xnorm,ynorm);

            diffis *= (const T &) cWeight(0);

            dxxgrad *= (const T &) cWeight(0);
            dyygrad *= (const T &) cWeight(0);

            dxnormgrad *= (const T &) cWeight(0);
            dynormgrad *= (const T &) cWeight(0);

            T dxyprod; dxyprod = 0.0;
            T dxnorm;  dxnorm  = 0.0;

            T dummy;

            dKKpro(dxyprod,dxnorm,res,xyprod,diffis,i,j,1,size()-1,xdim,2,dummy,dummy);

            xygrad = dxyprod*sxygrad;

            xnormgrad  = dxyprod*sxnormgrad;
            xnormgrad += dxnorm*dxxgrad;
            xnormgrad += dxnorm*dxnormgrad;
            xnormgrad += dxnorm*dxnormgrad;
        }

        else
        {
            int dummyind = 0;

            if ( xy10 )
            {
                xyprod = (*xy10);
                yxprod = (*xy10);

                xyprod += bias;
                yxprod += bias;
            }

            else if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
                yxprod = xyprod;
            }

            else
            {
                kernel8xx(0,xyprod,dummyind,cType(zeroint()),x,y,xinfo,yinfo,i,j,xdim,0,0,mlid);

                xyprod += bias;
                yxprod  = xyprod;
            }

            xyprod += yxprod;
            xyprod *= 0.5;

            if ( xy00 && xy11 )
            {
                diff2norm(diffis,(double) xyprod,(*xy00),(*xy11));
            }

            else if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else
            {
                T xnorm; xnorm = 0.0;
                T ynorm; ynorm = 0.0;

                int dummyind;

                kernel8xx(0,xnorm,dummyind,cType(zeroint()),x,x,xinfo,xinfo,i,i,xdim,0,0,mlid);
                kernel8xx(0,ynorm,dummyind,cType(zeroint()),y,y,yinfo,yinfo,j,j,xdim,0,0,mlid);

                if ( assumreal )
                {
                    diff2norm(diffis,(double) xyprod,(double) xnorm,(double) ynorm);
                }

                else
                {
                    diff2norm(diffis,(xyprod+yxprod)/2.0,xnorm,ynorm);
                }
            }

            xyprod *= (const T &) cWeight(0);
            diffis *= (const T &) cWeight(0);

            T dummy;

            dKKpro(xygrad,xnormgrad,res,xyprod,diffis,i,j,1,size()-1,xdim,2,dummy,dummy);
        }
    }

    else
    {
        NiceAssert( ismagterm == zeroint() );

        NiceAssert( deepDeriv );
        NiceAssert( ( isAltDiff() <= 199 ) || ( isAltDiff() >= 300 ) );

        gentype xyprod; xyprod = 0.0;
        gentype yxprod; yxprod = 0.0;

        gentype xnorm; xnorm = 0.0;
        gentype ynorm; ynorm = 0.0;

        if ( xy10 && xy00 && xy11 )
        {
            xyprod = (*xy10);
            yxprod = (*xy10);

            xyprod += bias;
            yxprod += bias;

            xnorm = (*xy00);
            ynorm = (*xy11);
        }

        else if ( pxyprod && pxyprod[0] )
        {
            xyprod = *pxyprod[0];
            yxprod = xyprod;

            xnorm = getmnorm(xinfo,x,2,xconsist,assumreal);
            ynorm = getmnorm(yinfo,y,2,xconsist,assumreal);
        }

        else if ( needsInner(-1,2) )
        {
            // This may be used by some kernels and not others, so calculate anyhow

            twoProductDiverted(xyprod,x,y,xconsist,assumreal);
            twoProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

            xyprod += bias;
            yxprod += bias;

            xnorm = getmnorm(xinfo,x,2,xconsist,assumreal);
            ynorm = getmnorm(yinfo,y,2,xconsist,assumreal);
        }

        else
        {
            xnorm = getmnorm(xinfo,x,2,xconsist,assumreal);
            ynorm = getmnorm(yinfo,y,2,xconsist,assumreal);
        }

        gentype tempres;
        gentype tempxygrad;
        gentype tempxnormgrad;

        dKdaz(tempxnormgrad,tempxygrad,minmaxind,xyprod,yxprod,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,mlid,assumreal);

        xygrad    = (T) tempxygrad;
        xnormgrad = (T) tempxnormgrad;
    }

    return;
}






//phantomx
template <class T>
void MercerKernel::d2LL2(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, 
                    const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                    const vecInfo &xinfo, const vecInfo &yinfo,
                    const T &bias,
                    const gentype **pxyprod,
                    int i, int j,
                    int xdim, int xconsist, int assumreal, int mlid, 
                    const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const
{
    T res;

    NiceAssert( !( isFastKernelSum() && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) );
    NiceAssert( !( isprod && !arexysimple(x,y) ) );

    minmaxind = -1;

    if ( isFastKernelSum() || isFastKernelChain() )
    {
        int needxxprod = isNormalised() || needsNorm();

        T xyprod; xyprod = 0.0;
        T yxprod; yxprod = 0.0;

        T xxprod; xxprod = 0.0;
        T yyprod; yyprod = 0.0;

        T diffis; diffis = 0.0;

        if ( xy10 )
        {
            xyprod = (*xy10);
            yxprod = (*xy10);

            xyprod += bias;
            yxprod += bias;
        }

        else if ( pxyprod && pxyprod[0] )
        {
            xyprod = *pxyprod[0];
            yxprod = xyprod;
        }

        else if ( needsInner(0,2) || ( isFastKernelSum() && needsInner(-1,2) ) )
        {
            twoProductDiverted(xyprod,x,y,xconsist,assumreal);
            twoProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

            xyprod += bias;
            yxprod += bias;
        }

        xyprod += yxprod;
        xyprod *= 0.5;

        if ( needxxprod )
        {
            if ( xy00 && xy11 )
            {
                xxprod = (*xy00);
                yyprod = (*xy11);

                xxprod += bias;
                yyprod += bias;
            }

            else
            {
                twoProductDiverted(xxprod,x,x,xconsist,assumreal);
                twoProductDiverted(yyprod,y,y,xconsist,assumreal);

                xxprod += bias;
                yyprod += bias;
            }
        }

        if ( ( needsDiff(0) || ( isFastKernelSum() && needsDiff() ) ) )
        {
            if ( xy00 && xy11 )
            {
                diff2norm(diffis,(double) xyprod,(*xy00),(*xy11));
            }

            else if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else
            {
                // Calculate ||x-y||^2 only as required

                if ( assumreal )
                {
                    diff2norm(diffis,(double) xyprod,(double) getmnorm(xinfo,x,2,xconsist,assumreal),(double) getmnorm(yinfo,y,2,xconsist,assumreal));
                }

                else
                {
                    diff2norm(diffis,(xyprod+yxprod)/2.0,getmnorm(xinfo,x,2,xconsist,assumreal),getmnorm(yinfo,y,2,xconsist,assumreal));
                }
            }
        }

        d2KKpro(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,res,xyprod,diffis,i,j,0,size()-1,xdim,2,xxprod,yyprod);
    }

    else if ( isFastKernelXfer() && !deepDeriv )
    {
        NiceAssert( ismagterm == zeroint() );

        T xyprod; xyprod = 0.0;
        T yxprod; yxprod = 0.0;
        T diffis; diffis = 0.0;

        if ( !( size() >= 2 ) || !needsDiff(1) )
        {
            int dummyind = 0;

            if ( xy10 )
            {
                xyprod = (*xy10);
                yxprod = (*xy10);

                xyprod += bias;
                yxprod += bias;
            }

            else if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
                yxprod = xyprod;
            }

            else
            {
                kernel8xx(0,xyprod,dummyind,cType(zeroint()),x,y,xinfo,yinfo,i,j,xdim,0,0,mlid);

                xyprod += bias;
                yxprod  = xyprod;
            }

            xyprod += yxprod;
            xyprod *= 0.5;

            xyprod *= (const T &) cWeight(0);

            T dummy; dummy = 0.0;

            d2KKpro(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,res,xyprod,diffis,i,j,1,size()-1,xdim,2,dummy,dummy);
        }

        else
        {
            int dummyind = 0;

            if ( xy10 )
            {
                xyprod = (*xy10);
                yxprod = (*xy10);

                xyprod += bias;
                yxprod += bias;
            }

            else if ( pxyprod && pxyprod[0] )
            {
                xyprod = *pxyprod[0];
                yxprod = xyprod;
            }

            else
            {
                kernel8xx(0,xyprod,dummyind,cType(zeroint()),x,y,xinfo,yinfo,i,j,xdim,0,0,mlid);

                xyprod += bias;
                yxprod  = xyprod;
            }

            xyprod += yxprod;
            xyprod *= 0.5;

            if ( xy00 && xy11 )
            {
                diff2norm(diffis,(double) xyprod,(*xy00),(*xy11));
            }

            else if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else
            {
                T xnorm; xnorm = 0.0;
                T ynorm; ynorm = 0.0;

                int dummyind;

                kernel8xx(0,xnorm,dummyind,cType(zeroint()),x,x,xinfo,xinfo,i,i,xdim,0,0,mlid);
                kernel8xx(0,ynorm,dummyind,cType(zeroint()),y,y,yinfo,yinfo,j,j,xdim,0,0,mlid);

                if ( assumreal )
                {
                    diff2norm(diffis,(double) xyprod,(double) xnorm,(double) ynorm);
                }

                else
                {
                    diff2norm(diffis,(xyprod+yxprod)/2.0,xnorm,ynorm);
                }
            }

            xyprod *= (const T &) cWeight(0);
            diffis *= (const T &) cWeight(0);

            T dummy;

            d2KKpro(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,res,xyprod,diffis,i,j,1,size()-1,xdim,2,dummy,dummy);
        }
    }

    else
    {
        throw("Second-order derivatives only implemented for simple cases");
    }

    return;
}





template <class T>
void MercerKernel::dnLL2del(Vector<T> &sc, Vector<Vector<int> > &nn, int &minmaxind, 
                           const Vector<int> &q, 
                           const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                           const vecInfo &xinfo, const vecInfo &yinfo, 
                           const T &bias, const gentype **pxyprod, 
                           int i, int j, 
                           int xdim, int xconsist, int assumreal, int mlid, 
                           const double *xy00, const double *xy10, const double *xy11, int deepDeriv) const
{
    (void) mlid;

    NiceAssert( ismagterm == zeroint() );

    NiceAssert( !( isFastKernelSum() && ( isAltDiff() >= 200 ) && ( isAltDiff() <= 299 ) ) );
    NiceAssert( !( isprod && !arexysimple(x,y) ) );

    (void) minmaxind;

    if ( isFastKernelSum() || isFastKernelChain() )
    {
//errstream() << "phantomzyza 0\n";
        // Evaluate all requires inner products

        T xyprod; xyprod = 0.0;
        T yxprod; yxprod = 0.0;
        T diffis; diffis = 0.0;

        if ( xy10 )
        {
            xyprod = (*xy10);
            yxprod = (*xy10);

            xyprod += bias;
            yxprod += bias;
        }

        else if ( pxyprod && pxyprod[0] )
        {
            xyprod = *pxyprod[0];
            yxprod = xyprod;
        }

        else if ( needsInner(0,2) || ( isFastKernelSum() && needsInner(-1,2) ) )
        {
            twoProductDiverted(xyprod,x,y,xconsist,assumreal);
            twoProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

            xyprod += bias;
            yxprod += bias;
        }

        xyprod += yxprod;
        xyprod *= 0.5;

        // Evaluate ||x-y||^2 if needed

        if ( ( needsDiff(0) || ( isFastKernelSum() && needsDiff() ) ) )
        {
            if ( xy00 && xy11 )
            {
                diff2norm(diffis,(double) xyprod,(*xy00),(*xy11));
            }

            else if ( pxyprod && pxyprod[1] )
            {
                diffis = *pxyprod[1];
            }

            else
            {
                // Calculate ||x-y||^2 only as required

                if ( assumreal )
                {
                    diff2norm(diffis,(double) xyprod,(double) getmnorm(xinfo,x,2,xconsist,assumreal),(double) getmnorm(yinfo,y,2,xconsist,assumreal));
                }

                else
                {
                    diff2norm(diffis,(xyprod+yxprod)/2.0,getmnorm(xinfo,x,2,xconsist,assumreal),getmnorm(yinfo,y,2,xconsist,assumreal));
                }
            }
        }




        // Gradient evaluate begins here

        int n = q.size();
        int z = 0;
        int ii,jj,k,l;

        // dnK/dx{q0}.dx{q1}... K(x,y) =  sum_i sc_i kronProd_{j=0,1,...} [ x{nn_ij}   if nn_ij = 0,1
        //                                                                [ kd{nn_ij}  if nn_ij < 0
        //
        // where: x{0} = x
        //        x{1} = y
        //        kd{a} kd{a} = kronecker-delta
        //
        // Method: initially compute nn,gd, where:
        //
        // gd(i) = [ g0 g1 g3 ]
        //
        // defines:
        //
        // sc(i) = K_{0,0,...,0,1,1,...,1,2,2,...,3}  (0 repeated g0 times, 1 g1 times, 2 g2 times)
        //
        // is a gradient of K wrt ||x||^2 (rn = 0), ||y||^2 (rn = 1), <x,y> (rn = 2)

//errstream() << "phantomzyza 1\n";
        Vector<Vector<int> > gd;

        sc.resize(1);
        nn.resize(1);
        gd.resize(1);

        sc("&",z) = 1.0;
        nn("&",z).resize(z);
        gd("&",z).resize(3);

        gd("&",z) = z;

        retVector<T>            tmpva;
        retVector<Vector<int> > tmpvb;
        
        for ( ii = n-1 ; ii >= 0 ; ii-- )
        {
//errstream() << "phantomzyza 1x: ii = " << ii << "\n";
            for ( jj = sc.size()-1 ; jj >= 0 ; jj-- )
            {
//errstream() << "phantomzyza 2: jj = " << jj << "\n";
                l = nn(jj).size();
                
//errstream() << "phantomzyza 2a: sc = " << sc << "\n";
//errstream() << "phantomzyza 2a: jj+1 = " << jj+1 << "\n";
//errstream() << "phantomzyza 2a: l+2 = " << l+2 << "\n";
                sc.addpad(jj+1,l+2);
//errstream() << "phantomzyza 2a: nn = " << nn << "\n";
//errstream() << "phantomzyza 2a: jj+1 = " << jj+1 << "\n";
//errstream() << "phantomzyza 2a: l+2 = " << l+2 << "\n";
                nn.addpad(jj+1,l+2);
//errstream() << "phantomzyza 2a: gd = " << gd << "\n";
//errstream() << "phantomzyza 2a: jj+1 = " << jj+1 << "\n";
//errstream() << "phantomzyza 2a: l+2 = " << l+2 << "\n";
                gd.addpad(jj+1,l+2);
                
//errstream() << "phantomzyza 2b\n";
                sc("&",jj+1,1,jj+l+2,tmpva) = sc(jj);
                nn("&",jj+1,1,jj+l+2,tmpvb) = nn(jj);
                gd("&",jj+1,1,jj+l+2,tmpvb) = gd(jj);
                
//errstream() << "phantomzyza 2c\n";
                for ( k = jj+l+2 ; k >= jj ; k-- )
                {
//errstream() << "phantomzyza 3: k = " << k << "\n";
//errstream() << "phantomzyza 3a: q = " << q << "\n";
//errstream() << "phantomzyza 3a: ii = " << ii << "\n";
//errstream() << "phantomzyza 3a: q(ii) = " << q(ii) << "\n";
//errstream() << "phantomzyza 3: k = " << k << "\n";
//errstream() << "phantomzyza 3: k-jj-3 = " << k-jj-3 << "\n";
//errstream() << "phantomzyza 3: nn = " << nn << "\n";
//errstream() << "phantomzyza 3: nn(k) = " << nn(k) << "\n";
                    if ( ( q(ii) == z ) && ( k == jj ) )
                    {
                        // d/dx, d/d||x||^2
                            
//errstream() << "phantomzyza 3b\n";
                        //gd("&",k).add(gd(k).size());
                        //gd("&",k)("&",gd(k).size()-1) = z;
                        //
                        //nn("&",k).add(nn(k).size());
                        //nn("&",k)("&",nn(k).size()-1) = z;

                        gd("&",k)("&",z)++;
                        
                        nn("&",k).add(z);
                        nn("&",k)("&",z) = z;

                        sc("&",k) *= 2.0;
                    }
                        
                    else if ( ( q(ii) == 1 ) && ( k == jj+1 ) )
                    {
                        // d/dy, d/d||y||^2
                            
//errstream() << "phantomzyza 3d\n";
                        //gd("&",k).add(gd(k).size());
                        //gd("&",k)("&",gd(k).size()-1) = 1;
                        //
                        //nn("&",k).add(nn(k).size());
                        //nn("&",k)("&",nn(k).size()-1) = 1;

                        gd("&",k)("&",1)++;
                        
                        nn("&",k).add(z);
                        nn("&",k)("&",z) = 1;
                        
                        sc("&",k) *= 2.0;
                    }
                        
                    else if ( ( q(ii) == z ) && ( k == jj+2 ) )
                    {
                        // d/dx, d/d<x,y>
                            
//errstream() << "phantomzyza 3c\n";
                        //gd("&",k).add(gd(k).size());
                        //gd("&",k)("&",gd(k).size()-1) = 2;
                        //
                        //nn("&",k).add(nn(k).size());
                        //nn("&",k)("&",nn(k).size()-1) = 1;

                        gd("&",k)("&",2)++;
                        
                        nn("&",k).add(z);
                        nn("&",k)("&",z) = 1;
                            
                        //sc("&",k) *= 1.0;
                    }
                        
                    else if ( ( q(ii) == 1 ) && ( k == jj+2 ) )
                    {
                        // d/dy, d/d<x,y>
                            
//errstream() << "phantomzyza 3e\n";
                        //gd("&",k).add(gd(k).size());
                        //gd("&",k)("&",gd(k).size()-1) = 2;
                        //
                        //nn("&",k).add(nn(k).size());
                        //nn("&",k)("&",nn(k).size()-1) = z;

                        gd("&",k)("&",2)++;
                        
                        nn("&",k).add(z);
                        nn("&",k)("&",z) = z;
                            
                        //sc("&",k) *= 1.0;
                    }
                    
                    else if ( ( k-jj-3 >= 0 ) && ( q(ii) == nn(k)(k-jj-3) ) )
                    {
//errstream() << "phantomzyza 3f\n";
                        //nn("&",k)("&",k-jj-3) = -ii;
                        //nn("&",k).add(k-jj-3);
                        //nn("&",k)("&",k-jj-3) = -ii;

                        nn("&",k)("&",k-jj-3) = -(ii+1);
                        nn("&",k).add(z);
                        nn("&",k)("&",z) = -(ii+1);
                            
                        //sc("&",k) *= 1.0;
                    }
                    
                    else
                    {
//errstream() << "phantomzyza 3g\n";
                        sc.remove(k);
                        nn.remove(k);
                        gd.remove(k);
                    }
//errstream() << "phantomzyza 3f\n";
                }
            }
        }
        
//errstream() << "phantomzyza 4\n";
//errstream() << "phantomzyza 4: sc = " << sc << "\n";
//errstream() << "phantomzyza 4: nn = " << nn << "\n";
//errstream() << "phantomzyza 4: gd = " << gd << "\n";

        // scratch-pad may be used by dnKKpro to pre-calculate on first call and re-use results later
        // (for example the RBF kernel calculates K(x,y) and stores it here as the derivatives are
        // simply scaled versions of this).

        T scratch;
        int isfirstcalc = 1;

        // gradpad keeps results for different kernel gradients to avoid later re-calculation

        SparseVector<SparseVector<SparseVector<T> > > gradpad;

        for ( k = nn.size()-1 ; k >= 0 ; k-- )
        {
//errstream() << "phantomzyza 5: k = " << k << "\n";
            if ( !(gradpad.isindpresent(gd(k)(z))) || !((gradpad(gd(k)(z))).isindpresent(gd(k)(1))) || !(((gradpad(gd(k)(z)))(gd(k)(1))).isindpresent(gd(k)(2))) )
            {
                dnKKpro(((gradpad("&",gd(k)(z)))("&",gd(k)(1)))("&",gd(k)(2)),gd(k),xyprod,diffis,i,j,0,size()-1,xdim,2,isfirstcalc,scratch);
                isfirstcalc = 0;
            }

//errstream() << "phantomzyza 6\n";
            
            sc("&",k) *= ((gradpad(gd(k)(z)))(gd(k)(1)))(gd(k)(2));

            if ( (double) abs2(sc(k)) == 0 )
            {
                sc.remove(k);
                nn.remove(k);
                gd.remove(k);
            }
//errstream() << "phantomzyza 7\n";
        }

        // Finally we do a quick, non-exhaustive scan of adjacent terms to see if
        // any can be combined.  gd is no longer relevant, so ignore that.  This
        // should actually catch most repeats due to the ordering applied previously.

        for ( k = nn.size()-1 ; k >= 1 ; k-- )
        {
            if ( nn(k) == nn(k-1) )
            {
                sc("&",k-1) += sc(k);

                sc.remove(k);
                nn.remove(k);
            }
        }
    }

    else if ( isFastKernelXfer() && !deepDeriv )
    {
        throw("High-order kernel transfer only implemented for simple cases");
    }

    else
    {
        throw("High-order derivative only implemented for simple cases");
    }

    return;
}
























#endif

