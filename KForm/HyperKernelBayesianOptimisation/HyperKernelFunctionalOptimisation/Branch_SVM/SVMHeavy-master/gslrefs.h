
// GSL abstraction layer
//
// TO DO: clear out these last few hold-outs
//
// Return value: 0 on success
//
// W0 is the standard W0 branch (W>-1) of the Lambert W function
// W1 is the W1 branch (W<-1) of the Lambert W function

#ifndef _gslrefs_h
#define _gslrefs_h

int gsl_dawson(double &res, double x);
int gsl_gamma_inc(double &res, double a, double x);
int gsl_psi(double &res, double x);
int gsl_psi_n(double &res, int n, double x);
int gsl_zeta(double &res, double x);
int gsl_lambertW0(double &res, double x);
int gsl_lambertW1(double &res, double x);

#endif

