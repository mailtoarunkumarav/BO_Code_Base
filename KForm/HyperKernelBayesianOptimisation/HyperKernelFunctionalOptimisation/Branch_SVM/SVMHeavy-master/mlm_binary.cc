
//
// Binary classification Type-II multi-layer kernel-machine
//
// Version: 7
// Date: 07/07/2018
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "mlm_binary.h"

// Boilerplate

MLM_Binary::MLM_Binary() : MLM_Generic()
{
    fixMLTree();

    thisthis = this;
    thisthisthis = &thisthis;

    return;
}

MLM_Binary::MLM_Binary(const MLM_Binary &src) : MLM_Generic()
{
    fixMLTree();

    thisthis = this;
    thisthisthis = &thisthis;

    assign(src,0);

    return;
}

MLM_Binary::MLM_Binary(const MLM_Binary &src, const ML_Base *srcx) : MLM_Generic()
{
    fixMLTree();

    thisthis = this;
    thisthisthis = &thisthis;

    assign(src,1);
    setaltx(srcx);

    return;
}

std::ostream &MLM_Binary::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary MLM\n\n";

    MLM_Generic::printstream(output,dep+1);

    return output;
}

std::istream &MLM_Binary::inputstream(std::istream &input )
{
    MLM_Generic::inputstream(input);

    return input;
}


// Actual stuff

// Noting the symmetry of the gradient matrices, and that we only ever
// need to access two of them, we cram all the matrices into one.

//#define DRDK(_q_,_i_,_j_)  dRdKstore(    ( _q_%2 ? ( ( ( _i_ <= _j_ ) ? _j_ : _i_ ) + 1 ) : ( ( _i_ <= _j_ ) ? _i_ : _j_ ) ),( _q_%2 ? ( ( _i_ <= _j_ ) ? _i_ : _j_ ) : ( ( ( _i_ <= _j_ ) ? _j_ : _i_ ) + 1 ) ))
//#define DRDKV(_q_,_i_,_j_) dRdKstore("&",( _q_%2 ? ( ( ( _i_ <= _j_ ) ? _j_ : _i_ ) + 1 ) : ( ( _i_ <= _j_ ) ? _i_ : _j_ ) ),( _q_%2 ? ( ( _i_ <= _j_ ) ? _i_ : _j_ ) : ( ( ( _i_ <= _j_ ) ? _j_ : _i_ ) + 1 ) ))

#define DRDK(_q_,_i_,_j_)  _q_%2 ? ( ( _i_ > _j_ ) ? dRdKstorea(_i_,_j_) : dRdKstorea(_j_,_i_) ) : ( ( _i_ > _j_ ) ? dRdKstoreb(_i_,_j_) : dRdKstoreb(_j_,_i_) )
#define DRDKV(_q_,_i_,_j_)  _q_%2 ? ( ( _i_ > _j_ ) ? dRdKstorea("&",_i_,_j_) : dRdKstorea("&",_j_,_i_) ) : ( ( _i_ > _j_ ) ? dRdKstoreb("&",_i_,_j_) : dRdKstoreb("&",_j_,_i_) )

int MLM_Binary::train(int &res, svmvolatile int &killSwitch)
{
    int locres = 0;

    if ( mltree.size() && N() )
    {
//        Matrix<double> dRdKstore(N()+1,N()+1);
        Matrix<double> dRdKstorea(N(),N());
        Matrix<double> dRdKstoreb(N(),N());
        Matrix<double> gradxy(N(),N());
        Matrix<double> gradxnorm(N(),N());
        Vector<double> gamma(N());
        Vector<double> chi(N());
        Vector<double> dRdbeta(N());
        Vector<double> beta(N());
        Vector<double> g;

        int isopt = 0;
        int locres = 0;
        int q,i,j,k,m,n,p; //,dummy;
        int Q = tsize();
        double lgst,Etot,Etotprev,gi,yi;
        double lr = mlmlr();
        double Ediffstop = diffstop()*N();
        double defspar = lsparse();

        // Randomise the hidden layers

//getQ().setQuadraticCost();
//getQ().setOptSMO();
errstream() << "Randomisation...";
        randomise(defspar);
errstream() << "done,";

// Try with fixed output layer
/*Vector<double> outweight(N());
for ( i = 0 ; i < N() ; i++ )
{
if ( ( (getQ().d())(i) == +1 ) || ( (getQ().d())(i) == +2 ) )
{
outweight("&",i) = +1.0;
}
else if ( (getQ().d())(i) == -1 )
{
outweight("&",i) = -1.0;
}
else
{
outweight("&",i) = 0.0;
}
}
errstream() << "phantomx 0: " << outweight << "\n";
errstream() << "phantomx 1: " << getQ().d() << "\n";
getQ().setAlphaR(outweight);*/

        // E(y,g) = max(0,1-gy)
        // dE/dg = 0  if yg >= 1
        //       = -y if yg < 1
        //
        // Alternative 1
        //
        // E(y,g) = 1 - logistic(gy)
        // logistic(x) = 1/(1+exp(-x))
        // dE/dg = -y.logistic(g).(1-logistic(g))
        //
        // Alternative 2
        //
        // E(y,g) = (1/2) (1 -logistic(gy))^2
        // logistic(x) = 1/(1+exp(-x))
        // dE/dg = -y.logistic(gy).(1-logistic(gy))^2

        Etotprev = -1;

//errstream() << "phantomx 0: starting\n";
        while ( !isopt )
        {
            // Forward pass via kernel reset

errstream() << "Forward pass";
            resetKernelTree();
errstream() << "\b\b\b\b\b\b\b\b\b\b\b\b";

            // Turnaround: train, output gradients, training error.

errstream() << "Turn-around...";
            locres |= getQ().train(res,killSwitch);
errstream() << "done,";
//errstream() << "phantomx 1: alpha = " << getQ().alphaR() << "\n";
//errstream() << "phantomx 1: Gp = " << getQ().Gp() << "\n";

            Etot = 0.0;

errstream() << "Turn-around II";
            for ( i = 0 ; i < N() ; i++ )
            {
                gentype tempresh,tempresg;

                //getQconst().gTrainingVector(gi,dummy,i);
                getQconst().ghTrainingVector(tempresh,tempresg,i);

                gi = (double) tempresg;

                yi = getQconst().d()(i);
//errstream() << "phantomx 2: i = " << i << ": gi = " << gi << "\n";
//errstream() << "phantomx 3: i = " << i << ": yi = " << yi << "\n";

                //lgst  = 1-(yi*gi);
                //lgst  = ( lgst > 0 ) ? lgst : 0;
                //Etot += lgst;
                //DRDKV(Q,i,i) = -yi*lgst;

                //lgst  = 1/(1+exp(-yi*gi));
                //Etot += 1-lgst;
                //DRDKV(Q,i,i) = -yi*lgst*(1-lgst);

                lgst  = 1/(1+exp(-yi*gi));
                Etot += (1-lgst)*(1-lgst);
                DRDKV(Q,i,i) = -yi*lgst*(1-lgst)*(1-lgst);

//errstream() << "phantomx 4: DRDKV: ";
                for ( j = 0 ; j <= i ; j++ )
                {
                    DRDKV(Q,i,j) = DRDK(Q,i,i)*(getQ().alphaR())(j);
//errstream() << DRDK(Q,i,j) << "\t";
                }
//errstream() << "\n";
            }
errstream() << "\b\b\b\b\b\b\b\b\b\b\b\b\b\b";

//errstream() << "phantomx 5: dE = " << Etotprev-Etot << "\n";
errstream() << Etot << " "; //phantomx
            if ( ( Etotprev >= 0.0 ) && ( Etotprev-Etot < Ediffstop ) && ( Etot < Etotprev ) )
            {
                isopt = 1;
            }

            Etotprev = Etot;

            if ( !isopt )
            {
                // Backward pass, weight update

                double dKqpdKqi,dKqpdKqj;

                for ( q = Q-1 ; q >= 0 ; q-- )
                {
//errstream() << "phantomx 8: q = " << q << "\n";
                    SVM_Generic &currlay = dynamic_cast<SVM_Generic &>(mltree("&",q));

                    // Calculate gradient matrices (K matrix is provided by caches in currlay.getKval(i,j))

errstream() << "dK calc";
                    for ( i = 0 ; i < N() ; i++ )
                    {
                        for ( j = i ; j < N() ; j++ )
                        {
                            currlay.dK(gradxy("&",i,j),gradxnorm("&",i,j),i,j);

                            gradxy("&",j,i)    = gradxy(i,j);
                            gradxnorm("&",j,i) = gradxnorm(i,j);
                        }
                    }
errstream() << "\b\b\b\b\b\b\b";
//errstream() << "phantomx 9: gradxy = " << gradxy << "\n";
//errstream() << "phantomx 10: gradxnorm = " << gradxnorm << "\n";

                    // grab beta

                    beta = currlay.alphaR();
//errstream() << "phantomx 11: beta = " << beta << "\n";

                    // Calculate gamma, chi matrices

                    gamma = 0.0;
                    chi   = 0.0;

errstream() << "gammachi";
                    for ( i = 0 ; i < N() ; i++ )
                    {
                        for ( p = 0 ; p < N() ; p++ )
                        {
                            gamma("&",i) += (currlay.getKval(i,p))*beta(p);
                            chi("&",i)   += gradxnorm(i,p)*beta(p);
                        }
                    }
errstream() << "\b\b\b\b\b\b\b\b";
//errstream() << "phantomx 12: gamma = " << gamma << "\n";
//errstream() << "phantomx 13: chi = " << chi << "\n";

                    // Calculate dR/dbeta

errstream() << "dRdbeta";
                    for ( k = 0 ; k < N() ; k++ )
                    {
                        if ( regtype(q) == 1 )
                        {
                            // 1-norm regularisation

                            dRdbeta("&",k) = (currlay.alphaState())(k);
                        }

                        else
                        {
                            // 2-norm regularisation

                            dRdbeta("&",k) = beta(k);
                        }

                        dRdbeta("&",k) /= regC(q);

                        for ( i = 0 ; i < N() ; i++ )
                        {
                            for ( j = i ; j < N() ; j++ )
                            {
                                dRdbeta("&",k) += ( ( j == k ) ? 1.0 : 2.0 )*DRDK(q+1,k,j)*( ((currlay.getKval(j,k))*gamma(i)) + ((currlay.getKval(k,i))*gamma(j)) );
                            }
                        }

                        dRdbeta("&",k) *= regC(q);
                    }
errstream() << "\b\b\b\b\b\b\b";
//errstream() << "phantomx 14: dRdbeta " << dRdbeta << "\n";

                    // Propogate dR/dK{q}

errstream() << "dKback";
                    if ( q > 0 )
                    {
                        for ( i = 0 ; i < N() ; i++ )
                        {
//errstream() << "phantomx 15: DRDK = ";
                            for ( j = i ; j < N() ; j++ )
                            {
                                // No need to worry about sym as this is stored as a triangular matrix

                                DRDKV(q,i,j) = 0.0;

                                for ( m = 0 ; m < N() ; m++ )
                                {
                                    for ( n = m ; n < N() ; n++ )
                                    {
                                        dKqpdKqi  = (0.5*gradxy(m,n)*((beta(m)*krondel(n,j))+(beta(n)*krondel(m,j))));
                                        dKqpdKqi += (gradxnorm(m,j)*beta(m)*krondel(m,n));
                                        dKqpdKqi += (chi(j)*krondel(m,j)*krondel(n,j));
                                        dKqpdKqi *= gamma(i);

                                        dKqpdKqj  = (0.5*gradxy(m,n)*((beta(m)*krondel(n,i))+(beta(n)*krondel(m,i))));
                                        dKqpdKqj += (gradxnorm(m,i)*beta(m)*krondel(m,n));
                                        dKqpdKqj += (chi(i)*krondel(m,i)*krondel(n,i));
                                        dKqpdKqj *= gamma(j);

                                        DRDKV(q,i,j) += ( ( m == n ) ? 1.0 : 2.0 )*( (dKqpdKqj*gamma(i)) + (dKqpdKqi*gamma(j)) );
                                    }
                                }
                            }
//errstream() << DRDK(q,i,j) << "\n";
                        }
//errstream() << "\n";
                    }
errstream() << "\b\b\b\b\b\b";

                    // Weight update

errstream() << "update";
                    for ( i = 0 ; i < N() ; i++ )
                    {
                        dRdbeta *= lr;

                        if ( regtype(q) == 1 )
                        {
                            // 1-norm regularisation

                            if ( (currlay.alphaState())(i) || ( dRdbeta(i) < -1 ) || ( dRdbeta(i) > +1 ) )
                            {
                                beta("&",i) -= dRdbeta(i);
                            }
                        }

                        else
                        {
                            beta("&",i) -= dRdbeta(i);
                        }
                    }
errstream() << "\b\b\b\b\b\b";

//errstream() << "phantomx 16: betaNew = " << beta << "\n\n\n";
                    currlay.setAlphaR(beta);
                }
            }
        }
    }

    else
    {
        locres |= getQ().train(res,killSwitch);
    }

    xistrained = 1;

    return locres;
}

