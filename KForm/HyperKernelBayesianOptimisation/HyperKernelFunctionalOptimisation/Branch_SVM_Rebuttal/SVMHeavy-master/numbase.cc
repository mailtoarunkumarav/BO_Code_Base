
//
// Non-standard functions for reals
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "numbase.h"
#include <math.h>
#include <stdlib.h>

int numbase_gamma(double &res, double x)
{
    if ( x <= 0.0 )
    {
        res = 0.0;
        return 1;
    }

    // Split the function domain into three intervals:
    // (0, 0.001), [0.001, 12), and (12, infinity)

    ///////////////////////////////////////////////////////////////////////////
    // First interval: (0, 0.001)
    //
    // For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
    // So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
    // The relative error over this interval is less than 6e-7.

    const double gamma = NUMBASE_EULER;

    if ( x < 0.001 )
    {
        res = 1.0/(x*(1.0+(gamma*x)));
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Second interval: [0.001, 12)
    
    if (x < 12.0)
    {
        // The algorithm directly approximates gamma over (1,2) and uses
        // reduction identities to reduce other arguments to this interval.
		
        double y = x;
        int n = 0;
        int arg_was_less_than_one = (y < 1.0);

        // Add or subtract integers as necessary to bring y into (1,2)
        // Will correct for this below

        if ( arg_was_less_than_one )
        {
            y += 1.0;
        }

        else
        {
            n =  ( (int) floor(y) ) - 1;  // will use n later
            y -= n;
        }

        // numerator coefficients for approximation over the interval (1,2)
        const static double p[] =
        {
            -1.71618513886549492533811E+0,
             2.47656508055759199108314E+1,
            -3.79804256470945635097577E+2,
             6.29331155312818442661052E+2,
             8.66966202790413211295064E+2,
            -3.14512729688483675254357E+4,
            -3.61444134186911729807069E+4,
             6.64561438202405440627855E+4
        };

        // denominator coefficients for approximation over the interval (1,2)
        const static double q[] =
        {
            -3.08402300119738975254353E+1,
             3.15350626979604161529144E+2,
            -1.01515636749021914166146E+3,
            -3.10777167157231109440444E+3,
             2.25381184209801510330112E+4,
             4.75584627752788110767815E+3,
            -1.34659959864969306392456E+5,
            -1.15132259675553483497211E+5
        };

        double num = 0.0;
        double den = 1.0;
        int i;

        double z = y - 1;

        for ( i = 0 ; i < 8 ; i++)
        {
            num = ( num + p[i] )*z;
            den = (den*z) + q[i];
        }

        double result = num/den + 1.0;

        // Apply correction if argument was not initially in (1,2)
        if (arg_was_less_than_one)
        {
            // Use identity gamma(z) = gamma(z+1)/z
            // The variable "result" now holds gamma of the original y + 1
            // Thus we use y-1 to get back the orginal y.
            result /= (y-1.0);
        }
        else
        {
            // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            for (i = 0; i < n; i++)
            {
                result *= y++;
            }
        }

        res = result;
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Third interval: [12, infinity)

    if (x > 171.624)
    {
        // Correct answer too large to display. Force +infinity.
        res = valpinf();
        return 2;
    }

    int retval = numbase_lngamma(res,x);
    res = exp(res);

    return retval;
}

int numbase_lngamma(double &res, double x)
{
    if ( x <= 0.0 )
    {
        res = 0.0;
        return 1;
    }

    if ( x < 12.0 )
    {
        int retval = numbase_gamma(res,x);
        res = log(fabs(res));

        return retval;
    }

    // Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252

    const static double c[8] =
    {
        1.0/12.0,
        -1.0/360.0,
        1.0/1260.0,
        -1.0/1680.0,
        1.0/1188.0,
        -691.0/360360.0,
        1.0/156.0,
        -3617.0/122400.0
    };

    double z = 1.0/(x*x);
    double sum = c[7];
    int i;

    for ( i = 6 ; i >= 0 ; i-- )
    {
        sum *= z;
        sum += c[i];
    }

    double series = sum/x;
    res = ( x - 0.5 )*log(x) - x + NUMBASE_HALFLOG2PI + series;

    return 0;
}


// Calculate volume of sphere of radius r in N dimensional space

double spherevol(double rsq, unsigned int N)
{
    double V = 0;

    // n even: vol = ((2.pi).r^2)^(n/2) / 2.4.....n
    //             = ((2.pi).r^2)^(n/2) / ((1.2.....n/2).(2^(n/2)))
    //             = prod_{i=1,2,...,n/2} (2.pi).r^2 / (2.i)
    //
    // n odd:  vol = (2.r).((2.pi).r^2)^((n-1)/2) / 1.3.....(n-1)
    //             = 2.r prod_{i=1,2,...,(n-1)/2} (2.pi).r^2 / (2.i+1)
    //
    // Note that rsq = r^2

    if ( N == 0 )
    {
        V = 1;
    }

    else if ( N == 1 )
    {
        V = 2*sqrt(rsq);
    }

    else if ( N == 2 )
    {
        V = NUMBASE_PI*rsq;
    }

    else if ( N > 2 )
    {
        unsigned int i;
        double Vx = 2*NUMBASE_PI*rsq;

        if ( N%2 )
        {
            // N odd

            V = 2*sqrt(rsq);

            for ( i = 1 ; i <= (N-1)/2 ; i++ )
            {
                V *= Vx/((2*i)+1);
            }
        }

        else
        {
            // N even

            V = 1;

            for ( i = 1 ; i <= N/2 ; i++ )
            {
                V *= Vx/(2*i);
            }
        }
    }

    return V;
}












// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------

// Bessel function code from somewhere on the web

/***********************************************************************
*                                                                      *
*    Program to calculate the first kind Bessel function of integer    *
*    order N, for any REAL X, using the function BESSJ(N,X).           *
*                                                                      *
* -------------------------------------------------------------------- *
*                                                                      *
*    SAMPLE RUN:                                                       *
*                                                                      *
*    (Calculate Bessel function for N=2, X=0.75).                      *
*                                                                      *
*    Bessel function of order  2 for X =  0.7500:                      *
*                                                                      *
*         Y =  0.06707400                                              *
*                                                                      *
* -------------------------------------------------------------------- *
*   Reference: From Numath Library By Tuan Dang Trong in Fortran 77.   *
*                                                                      *
*                               C++ Release 1.0 By J-P Moreau, Paris.  *
*                                        (www.jpmoreau.fr)             *
***********************************************************************/
//#include <math.h>
//#include <stdio.h>

double BESSJ0(double X);
double BESSJ1(double X);
double BESSJ(int N, double X);
double BESSY0(double X);
double BESSY1(double X);
double BESSY(int N, double X);
double BESSYP(int N, double X);


    double BESSJ0 (double X) {
/***********************************************************************
      This subroutine calculates the First Kind Bessel Function of
      order 0, for any real number X. The polynomial approximation by
      series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
      REFERENCES:
      M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
      C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
      VOL.5, 1962.
************************************************************************/
    const double
          P1=1.0, P2=-0.1098628627E-2, P3=0.2734510407E-4,
          P4=-0.2073370639E-5, P5= 0.2093887211E-6,
          Q1=-0.1562499995E-1, Q2= 0.1430488765E-3, Q3=-0.6911147651E-5,
          Q4= 0.7621095161E-6, Q5=-0.9349451520E-7,
          R1= 57568490574.0, R2=-13362590354.0, R3=651619640.7,
          R4=-11214424.18, R5= 77392.33017, R6=-184.9052456,
          S1= 57568490411.0, S2=1029532985.0, S3=9494680.718,
          S4= 59272.64853, S5=267.8532712, S6=1.0;
    double
          AX,FR,FS,Z,FP,FQ,XX,Y, TMP;

      if (X==0.0) return 1.0;
      AX = fabs(X);
      if (AX < 8.0) {
        Y = X*X;
        FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))));
        FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*S6))));
        TMP = FR/FS;
      }
      else {
        Z = 8./AX;
        Y = Z*Z;
        XX = AX-0.785398164;
        FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)));
        FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)));
        TMP = sqrt(0.636619772/AX)*(FP*cos(XX)-Z*FQ*sin(XX));
      }
      return TMP;
	}

    double Sign(double X, double Y);
    double Sign(double X, double Y) {
      if (Y<0.0) return (-fabs(X));
      else return (fabs(X));
    }

    double BESSJ1 (double X) {
/**********************************************************************
      This subroutine calculates the First Kind Bessel Function of
      order 1, for any real number X. The polynomial approximation by
      series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
      REFERENCES:
      M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
      C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
      VOL.5, 1962.
***********************************************************************/
    const double  
      P1=1.0, P2=0.183105E-2, P3=-0.3516396496E-4, P4=0.2457520174E-5,
      P5=-0.240337019E-6,  P6=0.636619772,
      Q1= 0.04687499995, Q2=-0.2002690873E-3, Q3=0.8449199096E-5,
      Q4=-0.88228987E-6, Q5= 0.105787412E-6,
      R1= 72362614232.0, R2=-7895059235.0, R3=242396853.1,
      R4=-2972611.439,   R5=15704.48260,  R6=-30.16036606,
      S1=144725228442.0, S2=2300535178.0, S3=18583304.74,
      S4=99447.43394,    S5=376.9991397,  S6=1.0;

	  double AX,FR,FS,Y,Z,FP,FQ,XX, TMP;

      AX = fabs(X);
      if (AX < 8.0) {
        Y = X*X;
        FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))));
        FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*S6))));
        TMP = X*(FR/FS);
      }
      else {
        Z = 8.0/AX;
        Y = Z*Z;
        XX = AX-2.35619491;
        FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)));
        FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)));
        TMP = sqrt(P6/AX)*(cos(XX)*FP-Z*sin(XX)*FQ)*Sign(S6,X);
      }
	  return TMP;
    }

    double BESSJ (int N, double X) {
/************************************************************************
      This subroutine calculates the first kind modified Bessel function
      of integer order N, for any REAL X. We use here the classical
      recursion formula, when X > N. For X < N, the Miller's algorithm
      is used to avoid overflows.
      ----------------------------- 
      REFERENCE:
      C.W.CLENSHAW, CHEBYSHEV SERIES FOR MATHEMATICAL FUNCTIONS,
      MATHEMATICAL TABLES, VOL.5, 1962.
*************************************************************************/
      const int IACC = 40; 
  	  const double BIGNO = 1e10,  BIGNI = 1e-10;
    
	  double TOX,BJM,BJ,BJP,SUM,TMP;
      int J, JSUM, M;

      if (N == 0) return BESSJ0(X);
      if (N == 1) return BESSJ1(X);
      if (X == 0.0) return 0.0;

      TOX = 2.0/X;
      if (X > 1.0*N) {
        BJM = BESSJ0(X);
        BJ  = BESSJ1(X);
        for (J=1; J<N; J++) {
          BJP = J*TOX*BJ-BJM;
          BJM = BJ;
          BJ  = BJP;
        }
        return BJ;
      }
      else {
        M = (int) (2*((N+floor(sqrt(1.0*(IACC*N))))/2));
        TMP = 0.0;
        JSUM = 0;
        SUM = 0.0;
        BJP = 0.0;
        BJ  = 1.0;
        for (J=M; J>0; J--) {
          BJM = J*TOX*BJ-BJP;
          BJP = BJ;
          BJ  = BJM;
          if (fabs(BJ) > BIGNO) {
            BJ  = BJ*BIGNI;
            BJP = BJP*BIGNI;
            TMP = TMP*BIGNI;
            SUM = SUM*BIGNI;
          }
          if (JSUM != 0)  SUM += BJ;
          JSUM = 1-JSUM;
          if (J == N)  TMP = BJP;
        }
        SUM = 2.0*SUM-BJ;
        return (TMP/SUM);
      }
    }


/***********************************************************************
*                                                                      *
*    Program to calculate the second kind Bessel function of integer   *
*    order N, for any REAL X, using the function BESSY(N,X).           *
*                                                                      *
* -------------------------------------------------------------------- *
*                                                                      *
*    SAMPLE RUN:                                                       *
*                                                                      *
*    (Calculate Bessel function for N=2, X=0.75).                      *
*                                                                      *
*    Second kind Bessel function of order  2 for X =  0.7500:          *
*                                                                      *
*         Y = -2.62974604                                              *
*                                                                      *
* -------------------------------------------------------------------- *
*   Reference: From Numath Library By Tuan Dang Trong in Fortran 77.   *
*                                                                      *
*                               C++ Release 1.0 By J-P Moreau, Paris.  *
*                                        (www.jpmoreau.fr)             *
***********************************************************************/
//#include <math.h>
//#include <stdio.h>

    double BESSY0 (double X) {
/* --------------------------------------------------------------------
      This subroutine calculates the Second Kind Bessel Function of
      order 0, for any real number X. The polynomial approximation by
      series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
      REFERENCES:
      M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
      C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
      VOL.5, 1962.
  --------------------------------------------------------------------- */
      const double
		  P1= 1.0, P2=-0.1098628627E-2, P3=0.2734510407E-4,
          P4=-0.2073370639E-5, P5= 0.2093887211E-6,
          Q1=-0.1562499995E-1, Q2= 0.1430488765E-3, Q3=-0.6911147651E-5,
          Q4= 0.7621095161E-6, Q5=-0.9349451520E-7,
          R1=-2957821389.0, R2=7062834065.0, R3=-512359803.6,
          R4= 10879881.29,  R5=-86327.92757, R6=228.4622733,
          S1= 40076544269.0, S2=745249964.8, S3=7189466.438,
          S4= 47447.26470,   S5=226.1030244, S6=1.0;
      double FS,FR,Z,FP,FQ,XX,Y;
 	  if (X == 0.0) return -1e30;
      if (X < 8.0) {
        Y = X*X;
        FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))));
        FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*S6))));
        return (FR/FS+0.636619772*BESSJ0(X)*log(X));
      }
      else {
        Z = 8.0/X;
        Y = Z*Z;
        XX = X-0.785398164;
        FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)));
        FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)));
        return (sqrt(0.636619772/X)*(FP*sin(XX)+Z*FQ*cos(XX)));
      }
    }

    double BESSY1 (double X) {
/* ---------------------------------------------------------------------
      This subroutine calculates the Second Kind Bessel Function of
      order 1, for any real number X. The polynomial approximation by
      series of Chebyshev polynomials is used for 0<X<8 and 0<8/X<1.
      REFERENCES:
      M.ABRAMOWITZ,I.A.STEGUN, HANDBOOK OF MATHEMATICAL FUNCTIONS, 1965.
      C.W.CLENSHAW, NATIONAL PHYSICAL LABORATORY MATHEMATICAL TABLES,
      VOL.5, 1962.
  ---------------------------------------------------------------------- */
    const double
      P1= 1.0, P2=0.183105E-2, P3=-0.3516396496E-4,
      P4= 0.2457520174E-5, P5=-0.240337019E-6,
      Q1= 0.04687499995, Q2=-0.2002690873E-3, Q3=0.8449199096E-5,
      Q4=-0.88228987E-6, Q5= 0.105787412E-6,
      R1=-0.4900604943E13, R2= 0.1275274390E13, R3=-0.5153438139E11,
      R4= 0.7349264551E9,  R5=-0.4237922726E7,  R6= 0.8511937935E4,
      S1= 0.2499580570E14, S2= 0.4244419664E12, S3= 0.3733650367E10,
      S4= 0.2245904002E8,  S5= 0.1020426050E6,  S6= 0.3549632885E3, S7=1.0;
    double  FR,FS,Z,FP,FQ,XX, Y;
      if (X == 0.0) return -1e30;
      if (X < 8.0) {
        Y = X*X;
        FR = R1+Y*(R2+Y*(R3+Y*(R4+Y*(R5+Y*R6))));
        FS = S1+Y*(S2+Y*(S3+Y*(S4+Y*(S5+Y*(S6+Y*S7)))));
        return (X*(FR/FS)+0.636619772*(BESSJ1(X)*log(X)-1.0/X));
      }
      else {
         Z = 8./X;
        Y = Z*Z;
        XX = X-2.356194491;
        FP = P1+Y*(P2+Y*(P3+Y*(P4+Y*P5)));
        FQ = Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*Q5)));
        return (sqrt(0.636619772/X)*(sin(XX)*FP+Z*cos(XX)*FQ));
      }
    }

	double BESSY (int N, double X) {
/* -----------------------------------------------------------------
      This subroutine calculates the second kind Bessel Function of
      integer order N, for any real X. We use here the classical
      recursive formula. 
  ------------------------------------------------------------------ */
    double TOX,BY,BYM,BYP; int J;
      if (N == 0) return BESSY0(X);
      if (N == 1) return BESSY1(X);
      if (X == 0.0) return -1e30;
      TOX = 2.0/X;
      BY  = BESSY1(X);
      BYM = BESSY0(X);
      for (J=1; J<N; J++) {
        BYP = J*TOX*BY-BYM;
        BYM = BY;
        BY  = BYP;
      };
      return BY;
	}
// --------------------------------------------------------------------------
    double BESSYP (int N, double X) {
      if (N == 0)
        return (-BESSY(1,X));
      else if (X == 0.0)
        return 1e-30;
      else
        return (BESSY(N-1,X)-(1.0*N/X)*BESSY(N,X));
    }

/***********************************************************************
*                                                                      *
*     Program to calculate the first kind modified Bessel function     *
*  of integer order N, for any REAL X, using the function BESSI(N,X).  *
*                                                                      *
* -------------------------------------------------------------------- *
*    SAMPLE RUN:                                                       *
*                                                                      *
*    (Calculate Bessel function for N=2, X=0.75).                      *
*                                                                      *
*    Bessel function of order 2 for X =  0.7500:                       *
*                                                                      *
*         Y = 0.073667                                                 *
*                                                                      *
* -------------------------------------------------------------------- *
*    Reference: From Numath Library By Tuan Dang Trong in Fortran 77.  *
*                                                                      *
*                               C++ Release 1.1 By J-P Moreau, Paris.  *
*                                        (www.jpmoreau.fr)             *
*                                                                      * 
*    Version 1.1: corected value of P4 in BESSIO (P4=1.2067492 and not *
*                 1.2067429) Aug. 2011.                                *
***********************************************************************/
#include <stdio.h>
#include <math.h>


  double BESSI0(double X);
  double BESSI1(double X);
  double BESSI(int N, double X);

// ---------------------------------------------------------------------
  double BESSI(int N, double X) {
/*----------------------------------------------------------------------
!     This subroutine calculates the first kind modified Bessel function
!     of integer order N, for any REAL X. We use here the classical
!     recursion formula, when X > N. For X < N, the Miller's algorithm
!     is used to avoid overflows. 
!     REFERENCE:
!     C.W.CLENSHAW, CHEBYSHEV SERIES FOR MATHEMATICAL FUNCTIONS,
!     MATHEMATICAL TABLES, VOL.5, 1962.
------------------------------------------------------------------------*/

      int IACC = 40; 
	  double BIGNO = 1e10, BIGNI = 1e-10;
      double TOX, BIM, BI, BIP, BSI;
      int J, M;

      if (N==0)  return (BESSI0(X));
      if (N==1)  return (BESSI1(X));
      if (X==0.0) return 0.0;

      TOX = 2.0/X;
      BIP = 0.0;
      BI  = 1.0;
      BSI = 0.0;
      M = (int) (2*((N+floor(sqrt(IACC*N)))));
      for (J = M; J>0; J--) {
        BIM = BIP+J*TOX*BI;
        BIP = BI;
        BI  = BIM;
        if (fabs(BI) > BIGNO) {
          BI  = BI*BIGNI;
          BIP = BIP*BIGNI;
          BSI = BSI*BIGNI;
        }
        if (J==N)  BSI = BIP;
      }
      return (BSI*BESSI0(X)/BI);
  }

// ----------------------------------------------------------------------
//  Auxiliary Bessel functions for N=0, N=1
  double BESSI0(double X) {
      double Y,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX;
      P1=1.0; P2=3.5156229; P3=3.0899424; P4=1.2067492;
      P5=0.2659732; P6=0.360768e-1; P7=0.45813e-2;
      Q1=0.39894228; Q2=0.1328592e-1; Q3=0.225319e-2;
      Q4=-0.157565e-2; Q5=0.916281e-2; Q6=-0.2057706e-1;
      Q7=0.2635537e-1; Q8=-0.1647633e-1; Q9=0.392377e-2;
      if (fabs(X) < 3.75) {
        Y=(X/3.75)*(X/3.75);
        return (P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7))))));
      }
      else {
        AX=fabs(X);
        Y=3.75/AX;
        BX=exp(AX)/sqrt(AX);
        AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))));
        return (AX*BX);
      }
  }

// ---------------------------------------------------------------------
  double BESSI1(double X) {
      double Y,P1,P2,P3,P4,P5,P6,P7,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,AX,BX;
      P1=0.5; P2=0.87890594; P3=0.51498869; P4=0.15084934;
      P5=0.2658733e-1; P6=0.301532e-2; P7=0.32411e-3;
      Q1=0.39894228; Q2=-0.3988024e-1; Q3=-0.362018e-2;
      Q4=0.163801e-2; Q5=-0.1031555e-1; Q6=0.2282967e-1;
      Q7=-0.2895312e-1; Q8=0.1787654e-1; Q9=-0.420059e-2;
      if (fabs(X) < 3.75) {
        Y=(X/3.75)*(X/3.75);
        return(X*(P1+Y*(P2+Y*(P3+Y*(P4+Y*(P5+Y*(P6+Y*P7)))))));
      }
      else {
        AX=fabs(X);
        Y=3.75/AX;
        BX=exp(AX)/sqrt(AX);
        AX=Q1+Y*(Q2+Y*(Q3+Y*(Q4+Y*(Q5+Y*(Q6+Y*(Q7+Y*(Q8+Y*Q9)))))));
        return (AX*BX);
      }
  }

static double bessk0( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function Kn(x) and n=0.  */
/*------------------------------------------------------------*/
{
   double y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(-log(x/2.0)*BESSI0(x))+(-0.57721566+y*(0.42278420
         +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
         +y*(0.10750e-3+y*0.74e-5))))));
   } else {
      y=2.0/x;
      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
         +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
         +y*(-0.251540e-2+y*0.53208e-3))))));
   }
   return ans;
}




static double bessk1( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function Kn(x) and n=1.  */
/*------------------------------------------------------------*/
{
   double y,ans;

   if (x <= 2.0) {
      y=x*x/4.0;
      ans=(log(x/2.0)*BESSI1(x))+(1.0/x)*(1.0+y*(0.15443144
         +y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
         +y*(-0.110404e-2+y*(-0.4686e-4)))))));
   } else {
      y=2.0/x;
      ans=(exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619
         +y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
         +y*(0.325614e-2+y*(-0.68245e-3)))))));
   }
   return ans;
}




/*
#>            bessk.dc2

Function:     bessk

Purpose:      Evaluate Modified Bessel function Kv(x) of integer order.

Category:     MATH

File:         bessel.c

Author:       M.G.R. Vogelaar

Use:          #include "bessel.h"
              double   result; 
              result = bessk( int n,
                              double x )


              bessk    Return the Modified Bessel function Kv(x) of 
                       integer order for input value x.
              n        Integer order of Bessel function.
              x        Double at which the function is evaluated.

                      
Description:  bessk evaluates at x the Modified Bessel function Kv(x) of 
              integer order n.
              This routine is NOT callable in FORTRAN.

Updates:      Jun 29, 1998: VOG, Document created.
#<
*/



double bessk( int n, double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function Kn(x) and n >= 0*/
/* Note that for x == 0 the functions bessy and bessk are not */
/* defined and a blank is returned.                           */
/*------------------------------------------------------------*/
{
   int j;
   double bk,bkm,bkp,tox;

NiceAssert( n >= 0 );
NiceAssert( x > 0 );

//   if (n < 0 || x == 0.0)
//   {
//      double   dblank;
//      setdblank_c( &dblank );
//      return( dblank );
//   }
   if (n == 0)
      return( bessk0(x) );
   if (n == 1)
      return( bessk1(x) );

   tox=2.0/x;
   bkm=bessk0(x);
   bk=bessk1(x);
   for (j=1;j<n;j++) {
      bkp=bkm+j*tox*bk;
      bkm=bk;
      bk=bkp;
   }
   return bk;
}









int numbase_j0(double &res, double x)
{ 
    if ( !base_j0(res,x) )
    {
        return 0;
    }

    res = BESSJ0(x);

    return 0;
}

int numbase_j1(double &res, double x) 
{ 
    if ( !base_j1(res,x) )
    {
        return 0;
    }

    res = BESSJ1(x);

    return 0;
}

int numbase_jn(double &res, int n, double x) 
{ 
    if ( !base_jn(res,n,x) )
    {
        return 0;
    }

    res = BESSJ(n,x);

    return 0;
}

int numbase_y0(double &res, double x)
{ 
    if ( !base_y0(res,x) )
    {
        return 0;
    }

    res = BESSY0(x);

    return 0;
}

int numbase_y1(double &res, double x) 
{ 
    if ( !base_y1(res,x) )
    {
        return 0;
    }

    res = BESSY1(x);

    return 0;
}

int numbase_yn(double &res, int n, double x) 
{ 
    if ( !base_yn(res,n,x) )
    {
        return 0;
    }

    res = BESSY(n,x);

    return 0;
}

int numbase_i0(double &res, double x)
{ 
    res = BESSI0(x);

    return 0;
}

int numbase_i1(double &res, double x) 
{ 
    res = BESSI1(x);

    return 0;
}

int numbase_in(double &res, int n, double x) 
{ 
    res = BESSI(n,x);

    return 0;
}

int numbase_k0(double &res, double x)
{ 
    res = bessk0(x);

    return 0;
}

int numbase_k1(double &res, double x) 
{ 
    res = bessk1(x);

    return 0;
}

int numbase_kn(double &res, int n, double x) 
{ 
    res = bessk(n,x);

    return 0;
}





















// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------

// Inverse error function code from somewhere on the web

#define erfinv_a3 -0.140543331
#define erfinv_a2 0.914624893
#define erfinv_a1 -1.645349621
#define erfinv_a0 0.886226899

#define erfinv_b4 0.012229801
#define erfinv_b3 -0.329097515
#define erfinv_b2 1.442710462
#define erfinv_b1 -2.118377725
#define erfinv_b0 1

#define erfinv_c3 1.641345311
#define erfinv_c2 3.429567803
#define erfinv_c1 -1.62490649
#define erfinv_c0 -1.970840454

#define erfinv_d2 1.637067800
#define erfinv_d1 3.543889200
#define erfinv_d0 1

inline int numbase_erfinv(double &res, double x)
{
    double x2,y;
    int sign_x;
 
    if ( ( x < -1 ) || ( x > 1 ) )
    {
        res = valvnan();

        return 1;
    }

    if ( x == 0 )
    {
        res = 0;

        return 0;
    }

    if ( x > 0 )
    {
        sign_x = 1;
    }

    else
    {
        sign_x = -1;
        x = -x;
    }

    if ( x <= 0.7 )
    {
        x2 = x*x;

        res  = x * (((erfinv_a3 * x2 + erfinv_a2) * x2 + erfinv_a1) * x2 + erfinv_a0);
        res /= (((erfinv_b4 * x2 + erfinv_b3) * x2 + erfinv_b2) * x2 + erfinv_b1) * x2 + erfinv_b0;
    }

    else
    {
        y  = sqrt (-log ((1 - x) / 2));

        res  = (((erfinv_c3 * y + erfinv_c2) * y + erfinv_c1) * y + erfinv_c0);
        res /= ((erfinv_d2 * y + erfinv_d1) * y + erfinv_d0);
    }

    res *= sign_x;
    x   *= sign_x;

    res -= ( erf(res) - x ) / (2 / sqrt(NUMBASE_PI) * exp(-res*res));
    res -= ( erf(res) - x ) / (2 / sqrt(NUMBASE_PI) * exp(-res*res));

    return 0;
}










