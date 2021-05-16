
//
// Battery simulation model
//
// See Cer1
//


class battSim;

// SOC       = state of charge at start of run
// THETA     = starting battery temperature, ambient temperature (degrees celcius)
//             (22.65 is chosen to match the results in Cer1)
// VCHARGED  = voltage when charged (2.4v chosen based on discussions with Hugh, but sims and wikipedia indicate 2.1V more realistic (maybe 2.4v is with current going in?))
// VDCHARGED = voltage when discharged (1.8v chosen based on wikipedia for full discharge - might need to revise as full discharge seems unlikely in-situ)

#define DEFAULT_SOC   1
#define DEFAULT_THETA 22.65

#define VCHARGED  2.1
#define VDCHARGED 1.8


parameters: 
battery params (a 21-d vector)
tmax
tdelta (battery runs from 0 to tmax, simulation interval tdelta - defaults are 3600 (an hour), 0.1 (seconds))
Vstart (target start voltage - default VCHARGED)

g(type>=0,tstart,tend,func) - overwrite tstart to tend with given function f(x).  By default we have current mode, 0 amps, all the time.  if tstart = 0 and voltage mode then sets start voltage
g(-1,v) - time to reach target voltage
g(-2,t,v) - t and v are vectors, returns how close the simulation is to the given data
g(-3,v) - v is a function, returns how close the simulation is to the given data
g(-4,dfile) - load datafile, initialise to ensure voltage matches at start, run simulation, return error between voltage start and voltage actual

train: actually runs the simulation

type: 216


class battSim
{
public:

    battSim(double _tdelta, double _theta = -1, double _Q_e = -1, double _SOC = -1)
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

        battStart(_tdelta,_theta,_Q_e,_SOC);

        return;
    }

    battSim(double _tdelta, double _theta, double _Q_e, double _SOC,
            double _C_0star, double _K_c, double _theta_f, double _epsilon, double _delta, double _Istar,
            double _E_m0, double _K_E, double _tau_1, double _R_00, double _R_10, double _A_0, double _R_20, double _A_21, double _A_22,
            double _E_p, double _V_p0, double _A_p, double _G_p0,
            double _C_theta, double _R_theta)
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

        battStart(_tdelta,_theta,_Q_e,_SOC);

        return;
    }

    void battStart(double _tdelta, double _theta = -1, double _Q_e = -1, double _SOC = -1)
    {
        // Steady state (I_1 ~ I_m), negligable parasitics (I ~ I_m)

        I   = 0; // Start with idle battery
        I_m = I;
        I_1 = I_m;

        theta   = ( _theta < 0 ) ? DEFAULT_THETA : _theta;
        theta_a = DEFAULT_THETA;

        if ( ( _Q_e < 0 ) && ( _SOC < 0 ) )
        {
            // Defaults set to make V = VCHARGED

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

                V_PN = E_m+(I_1*R_1)+(I_m*R_2);
                I_p  = V_PN*G_p0*exp((V_PN/V_p0)+(A_p*(1-(theta/theta_f))));
                R_p  = (V_PN-E_p)/I_p;

                I_m  = I-I_p;

                P_sp = I_p*I_p*R_p;
                P_r  = (R_0*I*I)+(R_1*I_1*I_1)+(R_2*I_m*I_m);

                P_s = P_sp+P_r;

                // Terminal

                I = I_m+I_p;
                V = E_m+(I*R_0)+(I_1*R_1)+(I_m*R_2);

//                std::cerr << V << ",";

                if ( V > VCHARGED )
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

            V_PN = E_m+(I_1*R_1)+(I_m*R_2);
            I_p  = V_PN*G_p0*exp((V_PN/V_p0)+(A_p*(1-(theta/theta_f))));
            R_p  = (V_PN-E_p)/I_p;

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
//        theta   = DEFAULT_THETA;
//        theta_a = DEFAULT_THETA;
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

    void Istep(int tsteps, double I_new, double theta_a_new)
    {
        int tn = t+tsteps;

        // Process as steady state up to change in current

        t++;

        for ( ; t <= tn ; t++ )
        {
            I = I_new;
            theta_a = theta_a_new;

            process_step();
        }

        return;
    }

    void Vstep(int tsteps, double V_new, double theta_a_new)
    {
        int tn = t+tsteps;

        // Process as steady state up to change in current

        t++;

        double P_new;

        for ( ; t <= tn ; t++ )
        {
            P_new = V_new*I;
            I = P_new/V;
            theta_a = theta_a_new;

            process_step();
        }

        return;
    }

    void Pstep(int tsteps, double P_new, double theta_a_new)
    {
        int tn = t+tsteps;

        // Process as steady state up to change in current

        t++;

        for ( ; t <= tn ; t++ )
        {
            I = P_new/V;
            theta_a = theta_a_new;

            process_step();
        }

        return;
    }

    // Process step

    void process_step(void)
    {
        double tstep = tdelta; // seconds?  hours (/(60*60))?;

        // pde

        Q_e   += -I_m*tstep;
        I_1   += ((I_m-I_1)/tau_1)*tstep;
        theta += (1/C_theta)*(P_s-((theta-theta_a)/R_theta))*tstep;

        // The rest

        SOC = 1-(Q_e/C(0,theta));
        DOC = 1-(Q_e/C(-I_1,theta));

        E_m = E_m0 - (K_E*(273+theta)*(1-SOC));
        R_0 = R_00*(1+(A_0*(1-SOC)));
        R_1 = -R_10*log(DOC);
        R_2 = R_20*exp(A_21*(1-SOC))/(1+exp(A_22*(I_m/Istar)));

        V_PN = E_m+(I_1*R_1)+(I_m*R_2);
        I_p  = V_PN*G_p0*exp((V_PN/V_p0)+(A_p*(1-(theta/theta_f))));
        R_p  = (V_PN-E_p)/I_p;

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


























//
// Battery simulation
//
// See Cer1
//

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include "battmodel.h"



// STEPSIZE  = simulation time-granularity (seconds)

#define STEPSIZE      1e-2



int main(int argc, const char *argv[])
{
    if ( argc != 6 )
    {
fallback:
        std::cerr << "Usage: comparperf.exe modelfile datfile m N s\n";
        std::cerr << "\n";
        std::cerr << "modelfile: C_0star (Ah)\n";
        std::cerr << "           K_c\n";
        std::cerr << "           theta_f\n";
        std::cerr << "           epsilon\n";
        std::cerr << "           delta\n";
        std::cerr << "           Istar\n";
        std::cerr << "     *     E_m0\n";;
        std::cerr << "           K_E\n";;
        std::cerr << "           tau_1\n";;
        std::cerr << "           R_00\n";;
        std::cerr << "           R_10\n";;
        std::cerr << "           R_20\n";;
        std::cerr << "           A_0\n";;
        std::cerr << "           A_21\n";;
        std::cerr << "           A_22\n";;
        std::cerr << "     *     E_p\n";;
        std::cerr << "           V_p0\n";;
        std::cerr << "           A_p\n";;
        std::cerr << "           G_p0\n";;
        std::cerr << "     *     C_theta\n";;
        std::cerr << "           R_theta\n";;
        std::cerr << "\n";
        std::cerr << "datfile: first line ignored, then on each line:\n";
        std::cerr << "           C1_Rec_ (iteration count, ignored)\n";
        std::cerr << "           C1_Tst_T_ (time (min) in simulation)\n";
        std::cerr << "           C1_Cur_A_ (current in or out, unsigned)\n";
        std::cerr << "           C1_Volt_V_ (terminal voltage)\n";
        std::cerr << "           C1_DPT_ (timestamp, ignored)\n";
        std::cerr << "           C1_Aux1_Tc_ (auxilliary temp 1 (ambient?))\n";
        std::cerr << "           C1_Aux2_Tc_ (auxilliary temp 1 (ambient?))\n";
        std::cerr << "           C1_zStage (state string, ignored)\n";
        std::cerr << "           C1_Stage_1 (1 for stage 1 charge - constant power C1_Cur_A_*C1_Volt_V_, 0 otherwise)\n";
        std::cerr << "           C1_Stage_2 (1 for stage 2 charge - constant voltageC1_Volt_V_, 0 otherwise)\n";
        std::cerr << "           C1_Stage_3 (1 for stage 3 charge - constant current C1_Cur_A_, 0 otherwise)\n";
        std::cerr << "\n";
        std::cerr << "m: startpoint in file\n";
        std::cerr << "N: number of datapoints to compare (-1 for all)\n";
        std::cerr << "s: Scalarisation\n";
        std::cerr << "\n";
        std::cerr << "Output: per line:\n";
        std::cerr << "           Time (seconds)\n";
        std::cerr << "           Vterm predicted (terminal voltage)\n";
        std::cerr << "           Vterm actual (terminal voltage)\n";
        std::cerr << "           Iin predicted (current in)\n";
        std::cerr << "           Iin actual (current in)\n";
        std::cerr << "           Battery temperature predicted\n";
        std::cerr << "           Mode (stage)\n";
        std::cerr << "\n";
        std::cerr << "std::cout: s*earlystop + ave_error\n";

        return 1;
    }

    const char *modelfname = argv[1];
    const char *datfname = argv[2];
    int m = atoi(argv[3]);
    int N = atoi(argv[4]);
    double s = atof(argv[5]);

    // Load model

    std::cerr << "Loading model file... ";

    std::ifstream modelfile(modelfname);

    if ( !modelfile.is_open() )
    {
        goto fallback;
    }

    double C_0star;
    double K_c;
    double theta_f;
    double epsilon;
    double delta;
    double Istar;

    double E_m0;
    double K_E;
    double tau_1;
    double R_00;
    double R_10;
    double R_20;
    double A_0;
    double A_21;
    double A_22;

    double E_p;
    double V_p0;
    double A_p;
    double G_p0;

    double C_theta;
    double R_theta;

    modelfile >> C_0star;
    modelfile >> K_c;
    modelfile >> theta_f;
    modelfile >> epsilon;
    modelfile >> delta;
    modelfile >> Istar;

    modelfile >> E_m0;
    modelfile >> K_E;
    modelfile >> tau_1;
    modelfile >> R_00;
    modelfile >> R_10;
    modelfile >> R_20;
    modelfile >> A_0;
    modelfile >> A_21;
    modelfile >> A_22;

    modelfile >> E_p;
    modelfile >> V_p0;
    modelfile >> A_p;
    modelfile >> G_p0;

    modelfile >> C_theta;
    modelfile >> R_theta;

    modelfile.close();

    std::cerr << "done.\n";

    // Count number of samples

    std::cerr << "Counting samples... ";

    std::ifstream datfilepre(datfname);

    if ( !datfilepre.is_open() )
    {
        goto fallback;
    }

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

    if ( !datfile.is_open() )
    {
        goto fallback;
    }

    double *Tset    = new double[numpts]; // time (seconds)
    double *Iin     = new double[numpts]; // current in (negative for discharge)
    double *Vterm   = new double[numpts]; // terminal voltage
    double *theta_a = new double[numpts]; // ambient temperature

    int *mode = new int[numpts]; // charge stage (1, 2 or 3 for charge, 0 for discharge)

    int i;

    double theta_a1;
    double theta_a2;

    int stage1;
    int stage2;
    int stage3;

    getline(datfile,dummy);
//std::cerr << dummy << "\t";

FIXME: call battStart to make sure voltages match up at beginning!

    for ( i = 0 ; i < numpts ; i++ )
    {
        datfile >> dummy;    // C1_Rec_ (iteration count, ignored)
//std::cerr << dummy << "\t";
        datfile >> Tset[i];  // C1_Tst_T_ (time (minutes) in simulation)
//std::cerr << Tset[i] << "\t";
        datfile >> Iin[i];   // C1_Cur_A_ (current in or out, unsigned)
//std::cerr << Iin[i] << "\t";
        datfile >> Vterm[i]; // C1_Volt_V_ (terminal voltage)
//std::cerr << Vterm[i] << "\t";
//        datfile >> dummy;    // C1_DPT_ (timestamp, ignored)
        std::getline(std::getline(datfile,skip,'"'),dummy, '"');
//std::cerr << dummy << "\t";
        datfile >> theta_a1; // C1_Aux1_Tc_ (auxilliary temp 1 (ambient?))
//std::cerr << theta_a1 << "\t";
        datfile >> theta_a2; // C1_Aux2_Tc_ (auxilliary temp 1 (ambient?))
//std::cerr << theta_a2 << "\t";
//        datfile >> dummy;    // C1_zStage (state string, ignored)
        std::getline(std::getline(datfile,skip,'"'),dummy, '"');
//std::cerr << dummy << "\t";
        datfile >> stage1;   // C1_Stage_1 (1 for stage 1 charge - constant power C1_Cur_A_*C1_Volt_V_, 0 otherwise)\n";
//std::cerr << stage1 << "\t";
        datfile >> stage2;   // C1_Stage_2 (1 for stage 2 charge - constant voltageC1_Volt_V_, 0 otherwise)\n";
//std::cerr << stage2 << "\t";
        datfile >> stage3;   // C1_Stage_3 (1 for stage 3 charge - constant current C1_Cur_A_, 0 otherwise)\n";
//std::cerr << stage3 << "\n";

        theta_a[i] = (theta_a1+theta_a2)/2;
        mode[i]    = stage1+(2*stage2)+(3*stage3);

        Tset[i] *= 60;
        Iin[i]  *= ( mode[i] ? 1 : -1 );
    }

    datfile.close();

    std::cerr << "done\n";

    // Run simulation and report

    std::cerr << "Start simulation:\n";

    double tdelta = STEPSIZE; // stepsize in temperature
    double thetaStart = theta_a[m]; // assuming we start with thermal equilibrium

    battSim simuBatt(tdelta,thetaStart,-1,-1,
                     C_0star,K_c,theta_f,epsilon,delta,Istar,
                     E_m0,K_E,tau_1,R_00,R_10,R_20,A_0,A_21,A_22,
                     E_p,V_p0,A_p,G_p0,
                     C_theta,R_theta);

    N = ( m+N > numpts ) ? numpts-m : N;

    int maxrunlen = N;
    int runlen = maxrunlen;
    int earlystop = 0;
    double Verr = 0.0;
    double Ierr = 0.0;

    for ( i = m ; i < m+N ; i++ )
    {
        double tdiff = i ? (Tset[i]-Tset[i-1]) : Tset[i];

        if ( mode[i] == 0 )
        {
            // Discharge

            simuBatt.Istep((int) (tdiff/STEPSIZE),Iin[i],theta_a[i]);
        }

        else if ( mode[i] == 1 )
        {
            // Charge stage 1 - constant power

            simuBatt.Pstep((int) (tdiff/STEPSIZE),Iin[i]*Vterm[i],theta_a[i]);
        }

        else if ( mode[i] == 2 )
        {
            // Charge stage 2 - constant voltage

            simuBatt.Vstep((int) (tdiff/STEPSIZE),Vterm[i],theta_a[i]);
        }

        else if ( mode[i] == 3 )
        {
            // Charge stage 3 - constant current

            simuBatt.Istep((int) (tdiff/STEPSIZE),Iin[i],theta_a[i]);
        }

        if ( ( runlen == maxrunlen ) && ( isnan(simuBatt.V) || isnan(simuBatt.I) || isnan(simuBatt.theta) ) )
        {
            runlen = i-m;
            earlystop = 1;

            break;
        }

        Verr += (Vterm[i]-simuBatt.V)*(Vterm[i]-simuBatt.V)/N;

        std::cout << Tset[i] << "\t" << Vterm[i] << "\t" << simuBatt.V << "\t" << Iin[i] << "\t" << simuBatt.I << "\t" << simuBatt.theta << "\t" << mode[i] << "\n";
    }

    if ( earlystop )
    {
        Verr *= ((double) N)/((double) (maxrunlen-runlen));
    }

    //std::cerr << maxrunlen-runlen << "\t" << Verr << "\n";
    std::cerr << (s*(maxrunlen-runlen))+Verr << "\n";

    // Clear memory and leave

    delete[] mode;

    delete[] Tset;
    delete[] Iin;
    delete[] Vterm;
    delete[] theta_a;

    return 0;
}
