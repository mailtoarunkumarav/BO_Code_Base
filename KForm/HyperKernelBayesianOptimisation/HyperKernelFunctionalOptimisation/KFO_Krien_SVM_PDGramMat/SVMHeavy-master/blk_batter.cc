//FIXME: add start temperature to ml_mutable etc

//
// Battery simulation block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include "blk_batter.h"

class battSim;


double runfullsim(battSim &simuBatt, const char *datfname, int m, int N, double s, int &ires, double battfixedTheta, int assumeIntExtSameTemp = 0);
double runfullsim(battSim &simuBatt, const Vector<gentype> &Tset, const Vector<gentype> &Iin, const Vector<gentype> &Vterm, const Vector<gentype> &theta_a, const Vector<gentype> &mode, int m, int N, double s, int &ires, double battfixedTheta, int numpts, int assumeIntExtSameTemp = 0);


BLK_Batter::BLK_Batter(int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    return;
}

BLK_Batter::BLK_Batter(const BLK_Batter &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_Batter::BLK_Batter(const BLK_Batter &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_Batter::~BLK_Batter()
{
    return;
}

std::ostream &BLK_Batter::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Battery simulation block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Batter::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}













































//
// Battery simulation model
//
// See Cer1
//

// SOC       = state of charge at start of run
// THETA     = starting battery temperature, ambient temperature (degrees celcius)
//             (22.65 is chosen to match the results in Cer1)
// VCHARGED  = voltage when charged (2.4v chosen based on discussions with Hugh, but sims and wikipedia indicate 2.1V more realistic (maybe 2.4v is with current going in?))
// VDCHARGED = voltage when discharged (1.8v chosen based on wikipedia for full discharge - might need to revise as full discharge seems unlikely in-situ)

#define DEFAULT_SOC       1
#define DEFAULT_BATTTHETA 22.65

#define VCHARGED  2.1
#define VDCHARGED 1.8

#define ITOL 0.1

class battSim
{
public:

    battSim(double _tdelta, double _theta = DEFAULT_BATTTHETA, double _Q_e = -1, double _SOC = -1, int _assumeDischarge = 0)
    {
        assert( ( _Q_e < 0 ) || ( _SOC < 0 ) );

        t = 0;
        tdelta = _tdelta;

        // Model parameters - see Cer1, battery 1

        C_0star = 261.9*60*60; // conversion Ah -> coulombs
        K_c     = 1.18;
        theta_f = -40;
        epsilon = 1.29;
        delta   = 1.40;
        Istar   = 49;

        E_m0  = 2.135;
        K_E   = 0.58e-3;
        tau_1 = 5000;
        R_00  = 2e-3;
        R_10  = 0.7e-3;
        R_20  = 15e-3;
        A_0   = -0.3;
        A_21  = -8;
        A_22  = -8.45;

        E_p  = 1.95;
        V_p0 = 0.1;
        A_p  = 2;
        G_p0 = 2e-12;

        C_theta = 15;
        R_theta = 0.2;

        assumeDischarge = _assumeDischarge;

        battStart(tdelta,_theta,VCHARGED,_Q_e,_SOC);

        return;
    }

    battSim(double Vstart, double _tdelta, double _theta, double _Q_e, double _SOC,
            double _C_0star, double _K_c, double _theta_f, double _epsilon, double _delta, double _Istar,
            double _E_m0, double _K_E, double _tau_1, double _R_00, double _R_10, double _A_0, double _R_20, double _A_21, double _A_22,
            double _E_p, double _V_p0, double _A_p, double _G_p0,
            double _C_theta, double _R_theta, 
            int _assumeDischarge)
    {
        assert( ( _Q_e < 0 ) || ( _SOC < 0 ) );

        t = 0;
        tdelta = _tdelta;

        // Model parameters - see Cer1, battery 1

        C_0star = _C_0star*60*60; // conversion Ah -> coulombs
        K_c     = _K_c;
        theta_f = _theta_f;
        epsilon = _epsilon;
        delta   = _delta;
        Istar   = _Istar;

        E_m0  = _E_m0;
        K_E   = _K_E;
        tau_1 = _tau_1;
        R_00  = _R_00;
        R_10  = _R_10;
        R_20  = _R_20;
        A_0   = _A_0;
        A_21  = _A_21;
        A_22  = _A_22;

        E_p  = _E_p;
        V_p0 = _V_p0;
        A_p  = _A_p;
        G_p0 = _G_p0;

        C_theta = _C_theta;
        R_theta = _R_theta;

        assumeDischarge = _assumeDischarge;

        battStart(tdelta,_theta,Vstart,_Q_e,_SOC);

        return;
    }

    void startVolt(double Vstart)
    {
        battStart(tdelta,theta,Vstart,-1,-1);

        return;
    }

    void battStart(double _tdelta, double _theta, double Vstart, double _Q_e = -1, double _SOC = -1)
    {
        // Steady state (I_1 ~ I_m), negligable parasitics (I ~ I_m)

        (void) _tdelta;

        theta   = _theta;
        theta_a = _theta;

        I   = 0; // Start with idle battery
        I_m = I;
        I_1 = I_m;

        if ( ( _Q_e < 0 ) && ( _SOC < 0 ) )
        {
            // Defaults set to make V = Vstart

            double Q_e_min = C(0,theta)*0.01;
            double Q_e_max = C(0,theta)*0.99;

            int done = 0;

            while ( !done )
            {
                Q_e = (Q_e_max+Q_e_min)/2;

                SOC = 1-(Q_e/C(0,theta));
                DOC = 1-(Q_e/C(-I_1,theta));

                E_m = E_m0 - (K_E*(273+theta)*(1-SOC));
                R_0 = R_00*(1+(A_0*(1-SOC)));
                R_1 = -R_10*log(DOC);
                R_2 = R_20*exp(A_21*(1-SOC))/(1+exp(A_22*(I_m/Istar)));

                // Sanity check - R_2 tends to go negative, despite the fact that this is impossible.

                R_1 = ( R_1 > 0 ) ? R_1 : 0.0;
                R_2 = ( R_2 > 0 ) ? R_2 : 0.0;

                V_PN = E_m+(I_1*R_1)+(I_m*R_2);
                I_p  = V_PN*G_p0*exp((V_PN/V_p0)+(A_p*(1-(theta/theta_f))));
                R_p  = (V_PN-E_p)/I_p;

                if ( assumeDischarge )
                {
                    R_2 = 0;
                    I_p = 0;
                }

                I_m  = I-I_p;

                P_sp = I_p*I_p*R_p;
                P_r  = (R_0*I*I)+(R_1*I_1*I_1)+(R_2*I_m*I_m);

                P_s = P_sp+P_r;

                // Terminal

                I = I_m+I_p;
                V = E_m+(I*R_0)+(I_1*R_1)+(I_m*R_2);

//                std::cerr << V << ",";

                if ( V > Vstart )
                {
                    Q_e_min = Q_e;
                }

                else
                {
                    Q_e_max = Q_e;
                }

                done = ( fabs(Q_e_max-Q_e_min) < 1e-3 ) ? 1 : 0;
            }
        }

        else if ( _SOC > 0 )
        {
            Q_e = (1-_SOC)*C(0,theta);

            SOC = 1-(Q_e/C(0,theta));
            DOC = 1-(Q_e/C(-I_1,theta));

            E_m = E_m0 - (K_E*(273+theta)*(1-SOC));
            R_0 = R_00*(1+(A_0*(1-SOC)));
            R_1 = -R_10*log(DOC);
            R_2 = R_20*exp(A_21*(1-SOC))/(1+exp(A_22*(I_m/Istar)));

            V_PN = E_m+(I_1*R_1)+(I_m*R_2);
            I_p  = V_PN*G_p0*exp((V_PN/V_p0)+(A_p*(1-(theta/theta_f))));
            R_p  = (V_PN-E_p)/I_p;

            if ( assumeDischarge )
            {
                R_2 = 0;
                I_p = 0;
            }

            I_m  = I-I_p;

            P_sp = I_p*I_p*R_p;
            P_r  = (R_0*I*I)+(R_1*I_1*I_1)+(R_2*I_m*I_m);

            P_s = P_sp+P_r;

            // Terminal

            I = I_m+I_p;
            V = E_m+(I*R_0)+(I_1*R_1)+(I_m*R_2);
        }

        else
        {
            Q_e = _Q_e;

            SOC = 1-(Q_e/C(0,theta));
            DOC = 1-(Q_e/C(-I_1,theta));

            E_m = E_m0 - (K_E*(273+theta)*(1-SOC));
            R_0 = R_00*(1+(A_0*(1-SOC)));
            R_1 = -R_10*log(DOC);
            R_2 = R_20*exp(A_21*(1-SOC))/(1+exp(A_22*(I_m/Istar)));

            // Sanity check - R_2 tends to go negative, despite the fact that this is impossible.

            R_1 = ( R_1 > 0 ) ? R_1 : 0.0;
            R_2 = ( R_2 > 0 ) ? R_2 : 0.0;

            V_PN = E_m+(I_1*R_1)+(I_m*R_2);
            I_p  = V_PN*G_p0*exp((V_PN/V_p0)+(A_p*(1-(theta/theta_f))));
            R_p  = (V_PN-E_p)/I_p;

            if ( assumeDischarge )
            {
                R_2 = 0;
                I_p = 0;
            }

            I_m  = I-I_p;

            P_sp = I_p*I_p*R_p;
            P_r  = (R_0*I*I)+(R_1*I_1*I_1)+(R_2*I_m*I_m);

            P_s = P_sp+P_r;

            // Terminal

            I = I_m+I_p;
            V = E_m+(I*R_0)+(I_1*R_1)+(I_m*R_2);
        }

//        // Battery state (reverse engineering Q_e from default SOC, assuming 0 < SOC << 1 for negligable parasitics)
//
//        theta   = DEFAULT_BATTTHETA;
//        theta_a = DEFAULT_BATTTHETA;
//
//        Q_e = (1-DEFAULT_SOC)*C(0,theta);
//
//        SOC = 1-(Q_e/C(0,theta));
//        DOC = 1-(Q_e/C(-I_1,theta));
//
//        E_m = E_m0 - (K_E*(273+theta)*(1-SOC));
//        R_0 = R_00*(1+(A_0*(1-SOC)));
//        R_1 = -R_10*log(DOC);
//        R_2 = R_20*exp(A_21*(1-SOC))/(1+exp(A_22*(I_m/Istar)));
//
//        V_PN = E_m+(I_1*R_1)+(I_m*R_2);
//        I_p  = V_PN*G_p0*exp((V_PN/V_p0)+(A_p*(1-(theta/theta_f))));
//        R_p  = (V_PN-E_p)/I_p;
//
//        I_m  = I-I_p;
//
//        P_sp = I_p*I_p*R_p;
//        P_r  = (R_0*I*I)+(R_1*I_1*I_1)+(R_2*I_m*I_m);
//
//        P_s = P_sp+P_r;
//
//        // Terminal
//
//        I = I_m+I_p;
//        V = E_m+(I*R_0)+(I_1*R_1)+(I_m*R_2);

        return;
    }

    // Operation
    //
    // - steady state for tsteps followed by step change to new current and
    //   ambient temperature.
    //
    // - current direction is into battery (so discharge is negative)
    // - ditto power: discharge is negative power, charge is positive

    void Istep(int tsteps, double I_new, double theta_a_new = -1000, int assumeIntExtSameTemp = 0)
    {
        int tn = t+tsteps;

        // Process as steady state up to change in current

        t++;

        for ( ; t <= tn ; t++ )
        {
            I = I_new;
            theta_a = ( theta_a_new > -1000 ) ? theta_a_new : theta_a;

            process_step(assumeIntExtSameTemp);
        }

        return;
    }

    void Vstep(int tsteps, double V_new, double theta_a_new = -1000, int assumeIntExtSameTemp = 0)
    {
        int tn = t+tsteps;

        // Process as steady state up to change in current

        t++;

        double P_new;

        for ( ; t <= tn ; t++ )
        {
            P_new = V_new*I;
            I = P_new/V;
            theta_a = ( theta_a_new > -1000 ) ? theta_a_new : theta_a;

            process_step(assumeIntExtSameTemp);
        }

        return;
    }

    void Pstep(int tsteps, double P_new, double theta_a_new = -1000, int assumeIntExtSameTemp = 0)
    {
        int tn = t+tsteps;

        // Process as steady state up to change in current

        t++;

        for ( ; t <= tn ; t++ )
        {
            I = P_new/V;
            theta_a = ( theta_a_new > -1000 ) ? theta_a_new : theta_a;

            process_step(assumeIntExtSameTemp);
        }

        return;
    }

    // Process step

    void process_step(int assumeIntExtSameTemp = 0)
    {
        double tstep = tdelta; // seconds?  hours (/(60*60))?;

        // pde

        Q_e   += -I_m*tstep;
        I_1   += ((I_m-I_1)/tau_1)*tstep;
        theta += (1/C_theta)*(P_s-((theta-theta_a)/R_theta))*tstep;

        // Simplifying assumption

        if ( assumeIntExtSameTemp )
        {
            NiceAssert( theta_a > -100 );

            theta = theta_a;
        }

        // The rest

        SOC = 1-(Q_e/C(0,theta));
        DOC = 1-(Q_e/C(-I_1,theta));

        E_m = E_m0 - (K_E*(273+theta)*(1-SOC));
        R_0 = R_00*(1+(A_0*(1-SOC)));
        R_1 = -R_10*log(DOC);
        R_2 = R_20*exp(A_21*(1-SOC))/(1+exp(A_22*(I_m/Istar)));

        // Sanity check - R_2 tends to go negative, despite the fact that this is impossible.

        R_1 = ( R_1 > 0 ) ? R_1 : 0.0;
        R_2 = ( R_2 > 0 ) ? R_2 : 0.0;

        V_PN = E_m+(I_1*R_1)+(I_m*R_2);
        I_p  = V_PN*G_p0*exp((V_PN/V_p0)+(A_p*(1-(theta/theta_f))));
        R_p  = (V_PN-E_p)/I_p;

        if ( assumeDischarge )
        {
            R_2 = 0;
            I_p = 0;
        }

        I_m  = I-I_p;

        P_sp = I_p*I_p*R_p;
        P_r  = (R_0*I*I)+(R_1*I_1*I_1)+(R_2*I_m*I_m);

        P_s = P_sp+P_r;

        V = E_m+(I*R_0)+(I_1*R_1)+(I_m*R_2);

        return;
    }





    // V = voltage at terminals
    // I = current into terminals
    // t = time (measured in deltas tdelta)
    //
    // tdelta = time increment

    double V;
    double I;

    int t;
    double tdelta;

    double I_1;
    double I_m;









    // Battery state - capacity and charge
    //
    // SOC = state of charge
    //     = 1 - Q_e/C(0,theta)
    // DOC = depth of charge
    //     = 1 - Q_e/C(I_1,theta)
    //
    // C(I,theta) = battery capacity
    //            = ( K_c C_0star ( 1 - theta/theta_f )^epsilon ) / ( 1 + ( K_c - 1 ).( I/Istar )^delta )
    //
    // I_1 = average current through R_1 in model
    // I_m = I-I_p (current *into* battery minus loss at parasitic branch)
    //
    // dQ_e/dt = -I_m
    // dI_1/dt = ( I_m - I_1)/tau_1
    //
    // tau_1 = R_1.C_1 (see main branch state)

    double Q_e;
    double SOC;
    double DOC;

    // Battery state - main branch
    // 
    // E_m = E_m0 - K_E.( 273 + theta ).( 1 - SOC )
    // R_0 = R_00 ( 1 + A_0.( 1 - SOC ) )
    // R_1 = -R_10.log(DOC)
    // R_2 = R_20 exp( A_21.( 1 - SOC ) )/( 1 + exp( A_22.(I_m/Istar) ) )

    double E_m;
    double R_0;
    double R_1;
    double R_2;

    // Battery state - parasitic branch
    //
    // I_p  = V_PN.G_p0.exp( (V_PN/V_p0) + Ap.( 1 - theta/theta_f ) )
    // V_PN = V - R_0.I
    //      = E_m + R_1.I_1 + R_2.I_m
    // R_p  = (V_PN-E_p)/I_p;

    double I_p;
    double V_PN;
    double R_p;

    // Battery state - thermal model
    //
    // theta   = electrolyte temperature
    // theta_a = ambient temperature
    // P_s     = source thermal power
    //
    // P_s = P_sp + P_r
    //
    // P_sp = R_p*I_p*I_p
    // P_r  = R_0.I*I + R_1.I_1*I_1 + R_2.I_m*I_m
    //
    // dtheta/dt = 1/C_theta ( P_s - (theta-theta_a)/R_theta )

    double theta;
    double theta_a;
    double P_s;
    double P_sp;
    double P_r;








    // Model parameters - battery capacity
    //
    // C_0star = battery capacity at 0 degrees celcius assuming nominal discharge current Istar
    // K_c     = ...
    // theta_f = freezing point of electrolyte in degrees celcius (typically -40)
    // epsilon = ...
    // delta   = ...
    // Istar   = nominal discharge current (typical current for average use)
    //           (alternatively, nominal capacity over nominal dicharge time)

    double C_0star;
    double K_c;
    double theta_f;
    double epsilon;
    double delta;
    double Istar;

    // Model parameters - main branch electrical equivalence

    double E_m0;
    double K_E;
    double tau_1;
    double R_00;
    double R_10;
    double A_0;
    double R_20;
    double A_21;
    double A_22;

    // Model parameters - parasitic branch reaction

    double E_p;
    double V_p0;
    double A_p;
    double G_p0;

    // Model parameters - thermal model
    //
    // C_theta = thermal capacity of battery
    // R_theta = thermal resistance b/w battery and environment

    double C_theta;
    double R_theta;





    // Assumptions
    //
    // assumeDischarge: if set then I_1 = R_2 = 0 automatically

    int assumeDischarge;




    // Calculate battery capacity

    double C(double I, double theta)
    {
//FIXME: it is entirely unclear what we should do here during a *charge*
//       cycle, which is required when calculating DOC.  Our work-around
//       (guess) is to set I = max(0,I) (ie. cut negatives to zero), but
//       this may be wrong.

        I = ( I <= 0.0 ) ? 0.0 : I;

        return (K_c*C_0star*pow(1-(theta/theta_f),epsilon))/(1+((K_c-1)*pow(I/Istar,delta)));
    }
};






int BLK_Batter::ghTrainingVector(gentype &resh, gentype &resg, int ijk, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    // g(3,func,v) - time to reach target voltage given battery charging with current = func(t) > 0
    // g(2,func,v) - time to reach target voltage given battery charging with voltage = func(t) > 0
    // g(1,func,v) - time to reach target voltage given battery charging with power = func(t) > 0
    // g(0,func,v) - time to drop to target voltage given battery discharging with current = func(t) > 0
    // g(-1,t,i,v) - given time (t), current (i) and voltage (v) vectors, assuming current charging, returns how close the simulation is to the given data (in terms of voltage)
    // g(-2,dfile) - given real battery data in datafile, return how colse the simulation is to the given data (in terms of voltage)
    //
    // return integer: 0 if all good, 1 if result is infinite, -1 if result is NaN

//errstream() << "phantomx 0: " << ijk << "\n";
    const SparseVector<gentype> &xx = x(ijk);
//errstream() << "phantomx 1: " << xx << "\n";

    double tdelta = batttdelta(); // stepsize in time
    double thetaStart = battthetaStart(); // start temperature (STC)
    double Vstart = battVstart();
    double Imax = battImax();

    int assumeDischarge = battneglectParasitic();
    double fixedTheta = battfixedTheta();

    double C_0star = battparam()(0);
    double K_c     = battparam()(1);
    double theta_f = battparam()(2);
    double epsilon = battparam()(3);
    double delta   = battparam()(4);
    double Istar   = battparam()(5);

    double E_m0  = battparam()(6);
    double K_E   = battparam()(7);
    double tau_1 = battparam()(8);
    double R_00  = battparam()(9);
    double R_10  = battparam()(10);
    double R_20  = battparam()(11);
    double A_0   = battparam()(12);
    double A_21  = battparam()(13);
    double A_22  = battparam()(14);

    double E_p  = battparam()(15);
    double V_p0 = battparam()(16);
    double A_p  = battparam()(17);
    double G_p0 = battparam()(18);

    double C_theta = battparam()(19);
    double R_theta = battparam()(20);

    battSim simuBatt(Vstart,tdelta,thetaStart,-1,-1,
                     C_0star,K_c,theta_f,epsilon,delta,Istar,
                     E_m0,K_E,tau_1,R_00,R_10,R_20,A_0,A_21,A_22,
                     E_p,V_p0,A_p,G_p0,
                     C_theta,R_theta,
                     assumeDischarge); //,
//                     battfixedTheta);

    int model = (int) xx(zeroint());
    int res = 0;
    double t;
    int i;

    switch ( model )
    {
        case 3:
        {
            // Time to v given charging current function

            gentype I = xx(1);  //errstream() << "phantomx 2: " << I << "\n";
            double Vtarget = (double) xx(2);
            double s = (double) xx(3);

            I.scalarfn_setisscalarfn(0);  //errstream() << "phantomx 3: " << I << "\n";

            for ( t = 0 ; ( t < batttmax() ) && ( simuBatt.V < Vtarget ) ; t += tdelta )
            {
                gentype tt(t/batttmax());
                gentype II = I(tt);  //errstream() << "phantomx 4: " << II << "\n";

                II.finalise(2);  //errstream() << "phantomx 5: " << II << "\n";
                II.finalise(1);  //errstream() << "phantomx 6: " << II << "\n";
                II.finalise();   //errstream() << "phantomx 7: " << II << "\n";

                double Ival = (double) II;  //errstream() << "phantomx 8: " << Ival << "\n";

                Ival = ( Ival < 0.0 ) ? 0.0 : ( ( Ival > Imax ) ? Imax : Ival );   //errstream() << "phantomx 9: " << Ival << "\n";

                simuBatt.Istep(1,Ival,fixedTheta);

/*
                int n = 0;
                std::ostringstream tmpbuff;
                tmpbuff << "I(" << t << ") =\t"     << Ival           << "\t" << simuBatt.V       << "\t" << simuBatt.I   << 
                                           "\tQS: " << simuBatt.Q_e   << "\t" << simuBatt.SOC     << "\t" << simuBatt.DOC << 
                                           "\tMB: " << simuBatt.E_m   << "\t" << simuBatt.R_0     << "\t" << simuBatt.R_1 << "\t" << simuBatt.R_2  << 
                                           "\tPB: " << simuBatt.I_p   << "\t" << simuBatt.V_PN    << "\t" << simuBatt.R_p << 
                                           "\tTB: " << simuBatt.theta << "\t" << simuBatt.theta_a << "\t" << simuBatt.P_s << "\t" << simuBatt.P_sp << 
                                           "\t";
                blankPrint(errstream(),n);
                n = nullPrint(errstream(),tmpbuff.str());
                errstream().flush();
*/

                if ( isnan(simuBatt.V) || isnan(simuBatt.I) || isnan(simuBatt.theta) )
                {
errstream() << "!!!" << t << "!!!";
                    res = 2;
                    break;
                }
            }

            outstream() << "\n";

            if ( !res && ( simuBatt.V < Vtarget ) )
            {
                res = 1;
            }

            if ( res == 0 )
            {
                // Reached target in finite time

                resg = t;
            }

            else if ( res == 1 )
            {
                // Failed to reach target

                resg = batttmax() + (s*(Vtarget-simuBatt.V));
            }

            else
            {
                // Model failure - treat as gone to inf

                resg = batttmax() + (s*Vtarget);
            }

            break;
        }

        case 2:
        {
            // Time to v given charging voltage function

            gentype V = xx(1);
            const double Vtarget = (double) xx(2);
            double s = (double) xx(3);

            V.scalarfn_setisscalarfn(0);

            for ( t = 0 ; ( t < batttmax() ) && ( simuBatt.V < Vtarget ) ; t += tdelta )
            {
                gentype tt(t/batttmax());
                gentype VV = V(tt);

                VV.finalise(2);
                VV.finalise(1);
                VV.finalise();

                double Vval = (double) VV;

                simuBatt.Vstep(1,Vval,fixedTheta);

                //int n = 0;
                //std::ostringstream tmpbuff;
                //tmpbuff << "V(" << t << ") = " << Vval << "\t" << simuBatt.V << "\t" << simuBatt.I << "\t" << simuBatt.theta << "\t" << simuBatt.theta_a;
                //blankPrint(errstream(),n);
                //n = nullPrint(errstream(),tmpbuff.str());
                //errstream().flush();

                if ( isnan(simuBatt.V) || isnan(simuBatt.I) || isnan(simuBatt.theta) )
                {
                    res = 2;
                    break;
                }
            }

            if ( !res && ( simuBatt.V < Vtarget ) )
            {
                res = 1;
            }

            if ( res == 0 )
            {
                // Reached target in finite time

                resg = t;
            }

            else if ( res == 1 )
            {
                // Failed to reach target

                resg = batttmax() + (s*(Vtarget-simuBatt.V));
            }

            else
            {
                // Model failure

                resg = batttmax() + (s*Vtarget);
            }

            break;
        }

        case 1:
        {
            // Time to v given charging power function

            gentype P = xx(1);
            const double Vtarget = (double) xx(2);
            double s = (double) xx(3);

            P.scalarfn_setisscalarfn(0);

            for ( t = 0 ; ( t < batttmax() ) && ( simuBatt.V < Vtarget ) ; t += tdelta )
            {
                gentype tt(t/batttmax());
                gentype PP = P(tt);

                PP.finalise(2);
                PP.finalise(1);
                PP.finalise();

                double Pval = (double) PP;

                simuBatt.Pstep(1,Pval,fixedTheta);

                //int n = 0;
                //std::ostringstream tmpbuff;
                //tmpbuff << "P(" << t << ") = " << Pval << "\t" << simuBatt.V << "\t" << simuBatt.I << "\t" << simuBatt.theta << "\t" << simuBatt.theta_a;
                //blankPrint(errstream(),n);
                //n = nullPrint(errstream(),tmpbuff.str());
                //errstream().flush();

                if ( isnan(simuBatt.V) || isnan(simuBatt.I) || isnan(simuBatt.theta) )
                {
                    res = 2;
                    break;
                }
            }

            if ( !res && ( simuBatt.V < Vtarget ) )
            {
                res = 1;
            }

            if ( res == 0 )
            {
                // Reached target in finite time

                resg = t;
            }

            else if ( res == 1 )
            {
                // Failed to reach target

                resg = batttmax() + (s*(Vtarget-simuBatt.V));
            }

            else
            {
                // Model failure

                resg = batttmax() + (s*Vtarget);
            }

            break;
        }

        case 0:
        {
            // Time to v given discharging current function

            gentype I = xx(1);
            const double Vtarget = (double) xx(2);
            double s = (double) xx(3);

            I.scalarfn_setisscalarfn(0);

            for ( t = 0 ; ( t < batttmax() ) && ( simuBatt.V > Vtarget ) ; t += tdelta )
            {
                gentype tt(t);
                gentype II = I(tt);

                II.finalise(2);
                II.finalise(1);
                II.finalise();

                double Ival = (double) II;

                Ival = ( Ival < 0.0 ) ? 0.0 : ( ( Ival > Imax ) ? Imax : Ival );

                simuBatt.Istep(1,-Ival,fixedTheta);

                //int n = 0;
                //std::ostringstream tmpbuff;
                //tmpbuff << "Idis(" << t << ") = " << Ival << "\t" << simuBatt.V << "\t" << simuBatt.I << "\t" << simuBatt.theta << "\t" << simuBatt.theta_a;
                //blankPrint(errstream(),n);
                //n = nullPrint(errstream(),tmpbuff.str());
                //errstream().flush();

                if ( isnan(simuBatt.V) || isnan(simuBatt.I) || isnan(simuBatt.theta) )
                {
                    res = 2;
                    break;
                }
            }

            if ( !res && ( simuBatt.V > Vtarget ) )
            {
                res = 1;
            }

            if ( res == 0 )
            {
                // Reached target in finite time

                resg = t;
            }

            else if ( res == 1 )
            {
                // Failed to reach target

                resg = batttmax() + (s*(Vtarget-simuBatt.V));
            }

            else
            {
                // Model failure

                resg = batttmax() + (s*Vtarget);
            }

            break;
        }

        case -1:
        {
            // Given time and current vectors, how close is model to reality

            const Vector<gentype> &tt = (const Vector<gentype> &) xx(1);
            const Vector<gentype> &ii = (const Vector<gentype> &) xx(2);
            const Vector<gentype> &vv = (const Vector<gentype> &) xx(3);
            double s = (double) xx(4);

            NiceAssert( tt.size() == ii.size() );
            NiceAssert( tt.size() == vv.size() );

            int N = tt.size();

            double dres = 0.0;

            for ( t = 0, i = 0 ; i < N ; i++ )
            {
                double tdiff = i ? ((double) (tt(i)-tt(i-1))) : ((double) tt(i));
                double Ival = (double) ii(i);
                double Vval = (double) vv(i);

                simuBatt.Istep((int) (tdiff/tdelta),Ival,fixedTheta);

                //std::ostringstream tmpbuff;
                //tmpbuff << "I(" << t << ") = " << Ival << "\t" << simuBatt.V << "\t" << simuBatt.I << "\t" << simuBatt.theta << "\t" << simuBatt.theta_a;
                //blankPrint(errstream(),n);
                //int n = nullPrint(errstream(),tmpbuff.str());
                //errstream().flush();

                if ( isnan(simuBatt.V) || isnan(simuBatt.I) || isnan(simuBatt.theta) )
                {
                    res = 2;
                    break;
                }

                dres += (Vval-simuBatt.V)*(Vval-simuBatt.V);
            }

            dres /= i;

            if ( res == 0 )
            {
                // Reached target in finite time

                resg = dres;
            }

            else
            {
                // Model failure

                resg = dres + (s*(batttmax()-t));
            }

            break;
        }

        case -2:
        {
            // Data in file

            const char *datfname = ((const std::string &) xx(1)).c_str();
            int m = (int) xx(2);
            int N = (int) xx(3);
            double s = (double) xx(4);

            resg = runfullsim(simuBatt,datfname,m,N,s,res,fixedTheta,0);

            break;
        }

        case -3:
        {
            // Data in vectors pre-loaded from file

errstream() << "phantomxqq 0\n";
            int m = (int) xx(1);
            int N = (int) xx(2);
            double s = (double) xx(3);
            const Vector<gentype> &Tset    = (const Vector<gentype> &) xx(4);
            const Vector<gentype> &Iin     = (const Vector<gentype> &) xx(5);
            const Vector<gentype> &Vterm   = (const Vector<gentype> &) xx(6);
            const Vector<gentype> &theta_a = (const Vector<gentype> &) xx(7);
            const Vector<gentype> &mode    = (const Vector<gentype> &) xx(8);

            resg = runfullsim(simuBatt,Tset,Iin,Vterm,theta_a,mode,m,N,s,res,fixedTheta,-1,0);

            break;
        }

        case -4:
        {
            // Data in file, assume interior temp = exterior temp

            const char *datfname = ((const std::string &) xx(1)).c_str();
            int m = (int) xx(2);
            int N = (int) xx(3);
            double s = (double) xx(4);

            resg = runfullsim(simuBatt,datfname,m,N,s,res,fixedTheta,1);

            break;
        }

        case -5:
        {
            // Data in vectors pre-loaded from file, assume interior temp = exterior temp

errstream() << "phantomxqq 0\n";
            int m = (int) xx(1);
            int N = (int) xx(2);
            double s = (double) xx(3);
            const Vector<gentype> &Tset    = (const Vector<gentype> &) xx(4);
            const Vector<gentype> &Iin     = (const Vector<gentype> &) xx(5);
            const Vector<gentype> &Vterm   = (const Vector<gentype> &) xx(6);
            const Vector<gentype> &theta_a = (const Vector<gentype> &) xx(7);
            const Vector<gentype> &mode    = (const Vector<gentype> &) xx(8);

            resg = runfullsim(simuBatt,Tset,Iin,Vterm,theta_a,mode,m,N,s,res,fixedTheta,-1,1);

            break;
        }

        default:
        {
            throw("Mode unknown in BLK_Batter evaluation");
            break;
        }
    }

    resh = resg;

    return res;
}











double runfullsim(battSim &simuBatt, const char *datfname, int m, int N, double s, int &ires, double battfixedTheta, int assumeIntExtSameTemp)
{
    // Count number of samples

    std::cerr << "Counting samples... ";

    std::ifstream datfilepre(datfname);

    NiceAssert( datfilepre.is_open() );

    int numpts = -1; // first line does not count
    std::string dummy;

    while ( getline(datfilepre,dummy) )
    {
//std::cerr << dummy << "\n";
        numpts++;
    }

    m = ( m < 0 ) ? 0 : m;
    N = ( N < 0 ) ? numpts : N;

    assert( m < numpts );

    datfilepre.close();

    std::cerr << "done (there are " << numpts << ")\n";

    // Load experiment data

    std::string skip;

    std::cerr << "Loading experimental results... ";

    std::ifstream datfile(datfname);

    NiceAssert( datfile.is_open() );

    Vector<gentype> Tset(numpts);    // time (seconds)
    Vector<gentype> Iin(numpts);     // current in (negative for discharge)
    Vector<gentype> Vterm(numpts);   // terminal voltage
    Vector<gentype> theta_a(numpts); // ambient temperature
    Vector<gentype> mode(numpts);    // charge stage (1, 2 or 3 for charge, 0 for discharge)

    int i;

    double theta_a1;
    double theta_a2;

    int stage1;
    int stage2;
    int stage3;

    getline(datfile,dummy);
//std::cerr << dummy << "\t";

    for ( i = 0 ; i < numpts ; i++ )
    {
        datfile >> dummy;        // C1_Rec_ (iteration count, ignored)                                                             //std::cerr << dummy << "\t";
        datfile >> Tset("&",i);  // C1_Tst_T_ (time (minutes) in simulation)                                                       //std::cerr << Tset[i] << "\t";
        datfile >> Iin("&",i);   // C1_Cur_A_ (current in or out, unsigned)                                                        //std::cerr << Iin[i] << "\t";
        datfile >> Vterm("&",i); // C1_Volt_V_ (terminal voltage)                                                                  //std::cerr << Vterm[i] << "\t";
        std::getline(std::getline(datfile,skip,'"'),dummy, '"');                                                               //std::cerr << dummy << "\t";
        datfile >> theta_a1;     // C1_Aux1_Tc_ (auxilliary temp 1 (ambient?))                                                     //std::cerr << theta_a1 << "\t";
        datfile >> theta_a2;     // C1_Aux2_Tc_ (auxilliary temp 1 (ambient?))                                                     //std::cerr << theta_a2 << "\t";
        std::getline(std::getline(datfile,skip,'"'),dummy, '"');                                                               //std::cerr << dummy << "\t";
        datfile >> stage1;       // C1_Stage_1 (1 for stage 1 charge - constant power C1_Cur_A_*C1_Volt_V_, 0 otherwise)\n";       //std::cerr << stage1 << "\t";
        datfile >> stage2;       // C1_Stage_2 (1 for stage 2 charge - constant voltageC1_Volt_V_, 0 otherwise)\n";                //std::cerr << stage2 << "\t";
        datfile >> stage3;       // C1_Stage_3 (1 for stage 3 charge - constant current C1_Cur_A_, 0 otherwise)\n";                //std::cerr << stage3 << "\n";

        theta_a("&",i) = (theta_a1+theta_a2)/2;
        mode("&",i)    = stage1+(2*stage2)+(3*stage3);

        Tset("&",i) *= 60.0;
        Iin("&",i)  *= ( ((int) mode(i)) ? 1.0 : -1.0 );
    }

    datfile.close();

    std::cerr << "done\n";

    return runfullsim(simuBatt,Tset,Iin,Vterm,theta_a,mode,m,N,s,ires,battfixedTheta,numpts,assumeIntExtSameTemp);
}






double runfullsim(battSim &simuBatt, const Vector<gentype> &Tset, const Vector<gentype> &Iin, const Vector<gentype> &Vterm, const Vector<gentype> &theta_a, const Vector<gentype> &mode, 
                  int m, int N, double s, int &ires, double battfixedTheta, int numpts, int assumeIntExtSameTemp)
{
    numpts = Tset.size();

    int i;

    // Run simulation and report

    std::cerr << "Start simulation (m = " << m << ", N = " << N << ", s = " << s << ", numpts = " << numpts << "):\n";

    N = ( m+N > numpts ) ? numpts-m : N;

    int maxrunlen = N;
    int runlen = maxrunlen;
    int earlystop = 0;
    double Verr = 0.0;
    double Ierr = 0.0;
    double tdelta = simuBatt.tdelta;

    (void) Ierr;

    int repperiod = 1000;

    simuBatt.startVolt(Vterm(m));

//errstream() << "phantomxyzqq 0 (" << m << "," << N << ")\n";
    for ( i = m ; i < m+N ; i++ )
    {
        double tdiff = i ? (((double) Tset(i))-((double) Tset(i-1))) : ((double) Tset(i));
        double thetaanow = ( battfixedTheta > -1000 ) ? battfixedTheta : ((double) theta_a(i));

//errstream() << tdiff/tdelta << " ";
        if ( ((int) mode(i)) == 0 )
        {
            // Discharge (Iin is negative for this step)

//errstream() << "D";
//errstream() << "phantomxyzqq 2\n";
            simuBatt.Istep((int) (tdiff/tdelta),((double) Iin(i)),thetaanow,assumeIntExtSameTemp);
        }

        else if ( ((int) mode(i)) == 1 )
        {
            // Charge stage 1 - constant power

//errstream() << "P";
//errstream() << "phantomxyzqq 3\n";
            simuBatt.Pstep((int) (tdiff/tdelta),((double) Iin(i))*((double) Vterm(i)),thetaanow,assumeIntExtSameTemp);
        }

        else if ( ((int) mode(i)) == 2 )
        {
            // Charge stage 2 - constant voltage

//errstream() << "V";
//errstream() << "phantomxyzqq 4\n";
            simuBatt.Vstep((int) (tdiff/tdelta),((double) Vterm(i)),thetaanow,assumeIntExtSameTemp);
        }

        else if ( ((int) mode(i)) == 3 )
        {
            // Charge stage 3 - constant current

//errstream() << "I";
//errstream() << "phantomxyzqq 5\n";
            simuBatt.Istep((int) (tdiff/tdelta),((double) Iin(i)),thetaanow,assumeIntExtSameTemp);
        }

//errstream() << "phantomxyzqq 6\n";
        int n = 0;
        if ( !(i%repperiod) )
        {
            std::ostringstream tmpbuff;
            tmpbuff << Tset(i) << "\t" << Vterm(i) << "\t" << simuBatt.V << "\t" << Iin(i) << "\t" << simuBatt.I << "\t" << simuBatt.theta << "\t" << ((int) mode(i));
            blankPrint(errstream(),n);
            n = nullPrint(errstream(),tmpbuff.str());
            errstream().flush();
        }

//errstream() << "phantomxyzqq 7\n";
        if ( ( runlen == maxrunlen ) && ( isnan(simuBatt.V) || isnan(simuBatt.I) || isnan(simuBatt.theta) ) )
        {
            errstream() << "Model failed at iteration " << i << "\n";

            runlen = i-m;
            earlystop = 1;

            break;
        }

//errstream() << "phantomxyzqq 8\n";
        Verr += (((double) Vterm(i))-simuBatt.V)*(((double) Vterm(i))-simuBatt.V)/N;
    }

    ires = 0;

    if ( earlystop )
    {
        ires = 1;
        Verr *= ((double) N)/((double) (maxrunlen-runlen));
    }

//errstream() << "phantomxyzqq 9\n";
    double res = (s*(maxrunlen-runlen))+Verr;

    //std::cerr << maxrunlen-runlen << "\t" << Verr << "\n";
    std::cerr << (s*(maxrunlen-runlen))+Verr << "\n";

    return res;
}
