
//
// Optimisation test functions as per wikipedia (see opttest.pdf)
//
// Version:
// Date: 1/12/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// fnnum values (acronyms as per DTLZ)
//
//  1: Rastrigin function
//  2: Ackley's function
//  3: Sphere function
//  4: Rosenbrock function
//  5: Beale's function
//  6: Goldstein–Price function
//  7: Booth's function
//  8: Bukin function N.6
//  9: Matyas function
// 10: Levi function N.13
// 11: Himmelblau's function
// 12: Three-hump camel function
// 13: Easom function
// 14: Cross-in-tray function
// 15: Eggholder function
// 16: Holder table function
// 17: McCormick function
// 18: Schaffer function N. 2
// 19: Schaffer function N. 4
// 20: Styblinski–Tang function


#include "opttest.h" 
#include "numbase.h" 
#include "matrix.h"
#include <math.h>

int evalTestFn(int fnnum, double &res, const Vector<double> &xx, const Matrix<double> *a)
{
    Vector<double> x(xx);

    int nonfeas = 0;
    int i;
    int n = x.size();
    double resshift = 0;
    double resscale = 1;

    res = 0;
    // at end: res = (res-resshift)*resscale);

    // Notes: fall through comments included to make gcc shutup about it

    switch ( fnnum )
    {
        case 1001:
        {
            x *= 5.12;
            resscale = 1.0/80.0;
            resshift = 0.0;
        }
        // fall through
        case 1:
        {
            // Rastrigin function

            double A = 10;

            res = A*n;

            if ( n )
            {
                for ( i = 0 ; i < n ; i++ )
                {
                    res += ((x(i)*x(i))-(A*cos(2*NUMBASE_PI*x(i))));
                }
            }

            break;
        }

        case 1002:
        {
            x *= 5.0;
            resscale = 1.0/15.0;
            resshift = 0.0;
        }
        // fall through
        case 2:
        {
            // Ackley's function

            if ( n )
            {
                for ( i = 0 ; i < n ; i++ )
                {
                    res += cos(2*NUMBASE_PI*x(i));
                }
            }

            res = -(20*exp(-0.2*sqrt(norm2(x)/n)))-exp(res/n)+NUMBASE_E+20;

            break;
        }

        case 1003:
        {
            x *= 2.0; // arbitrarily put range -2,2 following wikipedia graph
            resscale = 1.0/8.0;
            resshift = 0.0;
        }
        // fall through
        case 3:
        {
            // Sphere function

            res = norm2(x);

            break;
        }

        case 1004:
        {
            x *= 3.0; // see 1003
            resscale = 1.0/2500.0;
            resshift = 0.0;
        }
        // fall through
        case 4:
        {
            // Rosenbrock function

            NiceAssert( n > 1 );

            if ( n )
            {
                for ( i = 0 ; i < n ; i++ )
                {
                    res += ((100*(x(i+1)-x(i))*(x(i+1)-x(i)))+((x(i)-1)*(x(i)-1)));
                }
            }

            break;
        }

        case 1005:
        {
            x *= 4.5;
            resscale = 1.0/200000.0;
            resshift = 0.0;
        }
        // fall through
        case 5:
        {
            // Beale function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = (1.5-xx+(xx*yy))*(1.5-xx+(xx*yy));
            res += (2.25-xx+(xx*yy*yy))*(2.25-xx+(xx*yy*yy));
            res += (2.625-xx+(xx*yy*yy*yy))*(2.625-xx+(xx*yy*yy*yy));

            break;
        }

        case 1006:
        {
            x *= 2.0;
            resscale = 1.0/1000000.0;
            resshift = 3.0;
        }
        // fall through
        case 6:
        {
            // Goldstein–Price function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = (1+((xx+yy+1)*(xx+yy+1)*(19-(14*xx)+(3*xx*xx)-(14*yy)+(6*xx*yy)+(3*yy*yy))));
            res *= (30+(((2*xx)-(3*yy))*((2*xx)-(3*yy))*(18-(32*xx)+(12*xx*xx)+(48*yy)-(26*xx*yy)+(27*yy*yy))));

            break;
        }

        case 1007:
        {
            x *= 10.0;
            resscale = 1.0/2500.0;
            resshift = 0.0;
        }
        // fall through
        case 7:
        {
            // Booth's function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = ((xx+(2*yy)-7)*(xx+(2*yy)-7));
            res += (((2*xx)+yy-5)*((2*xx)+yy-5));

            break;
        }

        case 1008:
        {
            x *= 3.0;
            resscale = 1.0/250.0;
            resshift = 0.0;
        }
        // fall through
        case 8:
        {
            // Bukin function N.6

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = (100*sqrt(abs2(yy-(0.01*xx*xx)))) + (0.01*abs2(xx+10));

            break;
        }

        case 1009:
        {
            x *= 10.0;
            resscale = 1.0/100.0;
            resshift = 0.0;
        }
        // fall through
        case 9:
        {
            // Matyas function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = (0.26*((xx*xx)+(yy*yy)))-(0.48*xx*yy);

            break;
        }

        case 1010:
        {
            x *= 10.0;
            resscale = 1.0/450.0;
            resshift = 0.0;
        }
        // fall through
        case 10:
        {
            // Levi function N.13

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = (sin(3*NUMBASE_PI*xx)*sin(3*NUMBASE_PI*xx));
            res += (xx-1)*(xx-1)*(1+(sin(3*NUMBASE_PI*yy)*sin(3*NUMBASE_PI*yy)));
            res += (yy-1)*(yy-1)*(1+(sin(2*NUMBASE_PI*yy)*sin(2*NUMBASE_PI*yy)));

            break;
        }

        case 1011:
        {
            x *= 5.0;
            resscale = 1.0/2000.0;
            resshift = 0.0;
        }
        // fall through
        case 11:
        {
            // Himmelblau's function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = (((xx*xx)+yy-11)*((xx*xx)+yy-11));
            res += ((xx+(yy*yy)-7)*(xx+(yy*yy)-7));

            break;
        }

        case 1012:
        {
            x *= 5.0;
            resscale = 1.0/2000.0;
            resshift = 0.0;
        }
        // fall through
        case 12:
        {
            // Three-hump camel function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = (2*xx*xx)-(1.05*xx*xx*xx*xx)+(xx*xx*xx*xx*xx*xx/6)+(xx*yy)+(yy*yy);

            break;
        }

        case 1013:
        {
            x *= 100.0;
            resscale = 1.0;
            resshift = -1.0;
        }
        // fall through
        case 13:
        {
            // Easom function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = -cos(xx)*cos(yy)*exp(-((xx-NUMBASE_PI)*(xx-NUMBASE_PI))-((yy-NUMBASE_PI)*(yy-NUMBASE_PI)));

            break;
        }

        case 1014:
        {
            x *= 10.0;
            resscale = 2.0;
            resshift = -2.06261;
        }
        // fall through
        case 14:
        {
            // Cross-in-tray function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = -0.0001*pow(abs2(sin(xx)*sin(yy)*exp(abs2(100-(abs2(x)/NUMBASE_PI))))+1,0.1);

            break;
        }

        case 1015:
        {
            x *= 512.0;
            resscale = 1.0/2000.0;
            resshift = -959.6407;
        }
        // fall through
        case 15:
        {
            // Eggholder function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = -(yy+47)*sin(sqrt(abs2((xx/2)+yy+47)));
            res -= xx*sin(sqrt(abs2(xx-(yy+47))));

            break;
        }

        case 1016:
        {
            x *= 10.0;
            resscale = 1.0/20.0;
            resshift = -19.2085;
        }
        // fall through
        case 16:
        {
            // Holder table function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = -abs2(sin(xx)*cos(yy)*exp(abs2(1-(abs2(x)/NUMBASE_PI))));

            break;
        }

        case 1017:
        {
            x *= 3.0;
            resscale = 1.0/42.0;
            resshift = -1.9133;
        }
        // fall through
        case 17:
        {
            // McCormick function

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = sin(xx+yy)+((xx-yy)*(xx-yy))-(1.5*xx)+(2.5*yy)+1;

            break;
        }

        case 1018:
        {
            x *= 100.0;
            resscale = 1.0;
            resshift = 0.0;
        }
        // fall through
        case 18:
        {
            // Schaffer function N. 2

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = 0.5+(((sin((xx*xx)-(yy*yy))*sin((xx*xx)-(yy*yy)))-0.5)/((1+(0.001*((xx*xx)+(yy*yy))))*(1+(0.001*((xx*xx)+(yy*yy))))));

            break;
        }

        case 1019:
        {
            x *= 100.0;
            resscale = 1.0/0.7;
            resshift = 0.292579;
        }
        // fall through
        case 19:
        {
            // Schaffer function N. 4

            NiceAssert( n == 2 );

            double xx = x(zeroint());
            double yy = x(1);

            res  = 0.5+(((cos(sin((xx*xx)-(yy*yy)))*cos(sin((xx*xx)-(yy*yy))))-0.5)/((1+(0.001*((xx*xx)+(yy*yy))))*(1+(0.001*((xx*xx)+(yy*yy))))));

            break;
        }

        case 1020:
        {
            x *= 5.0;
            resscale = 1.0/250.0; // this is probably wrong
            resshift = -39.16616*x.size();
        }
        // fall through
        case 20:
        {
            // Styblinski–Tang function

            if ( n )
            {
                for ( i = 0 ; i < n ; i++ )
                {
                    res += (x(i)*x(i)*x(i)*x(i))-(16*x(i)*x(i))+(5*x(i));
                }

                res /= 2;
            }

            break;
        }

        case 1021:
        {
            x *= 2.0;
            x -= 1.0;
            resscale = 1.0; // nominal
            resshift = -1.3;
        }
        // fall through
        case 21:
        {
            // Stability test function 1

            NiceAssert( n == 1 );

            double xx = x(zeroint());

            res = exp(-20*(xx-0.2)*(xx-0.2))+exp(-20*sqrt(0.00001+((xx-0.5)*(xx-0.5))))+exp(2*(xx-0.8));

            break;
        }

        case 1022:
        {
            x *= 2.0;
            x -= 1.0;
            resscale = 1.0; // nominal
            resshift = -1.0;
        }
        // fall through
        case 22:
        {
            // Stability test function 2

            NiceAssert( n == 1 );

            double xx = x(zeroint());
            static double gamma = 1/(5*sqrt(2));

            res = (4*exp(-(xx-1)*(xx-1)/(2*gamma*gamma))) + exp(-(xx-0.5)*(xx-0.5)/(2*gamma*gamma));

            break;
        }

        case 1023:
        {
            x *= 2.0;
            x -= 1.0;
            resscale = 1.0; // nominal
            resshift = 0.0; // nominal
        }
        // fall through
        case 23:
        {
            // 23: Test function 3: f(x) = sum_i a_{i,0} exp(-||x-x_{i,2:...}||_2^2/(2*a_{i,1}*a_{i,1}))

            NiceAssert( a );

            const Matrix<double> &aa = *a;

            NiceAssert( n = aa.numCols()-2 );

            int m = aa.numRows();

            res = 0;

            if ( n && m )
            {
                for ( i = 0 ; i < m ; i++ )
                {
                    retVector<double> tmpva;

                    double alpha = aa(i,zeroint());
                    double gamma = aa(i,1);

                    const Vector<double> &xa = aa(i,2,1,2+n-1,tmpva);

                    res += alpha*exp(-norm2(x-xa)/(2*gamma*gamma));
                }
            }

            break;
        }

        default:
        {
            res = 0;

            break;
        }
    }

//OLD VERSION (WHICH ALSO HAD ALL resshifts above negated):    res = (res*resscale)-resshift;
    res = (res-resshift)*resscale;

    return nonfeas;
}

