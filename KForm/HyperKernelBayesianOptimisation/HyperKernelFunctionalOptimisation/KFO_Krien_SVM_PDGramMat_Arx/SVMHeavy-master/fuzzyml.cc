
//
// Fuzzy weight selection for MLs
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "fuzzyml.h"
#include "svm_single.h"

double calcK(int samekern, const ML_Base &ml, const MercerKernel &distkern, int i, int j);

int calcFuzzML(ML_Base &ml, const gentype &fuzzfn, const SparseVector<SparseVector<gentype> > &argvariables, const MercerKernel &distkern, double f, double m, double nu, int setCoreps)
{
    int res = 0;
    int i,j,k,l;
    //int dummya,dummyb = 0;
    int samekern = ( ml.getKernel() == distkern );

    // Method:
    //
    // - Calculate an index vector for training examples in each class.
    // - Calculate distance of each point from all class means
    // - Calculate distances between class means
    // - Calculate sphere radii for each class
    // - If fuzzfn uses var(2,6) or var(2,10), implement 1-class SVM for
    //   each class and calculate outputs for each point.  Otherwise just
    //   set this to 1.
    // - Calculate q1,q2,q3,q4 for each point.
    // - Use fuzzfn to calculate C/epsilon weights.
    // - Apply C/epsilon weights to the ML.
    //
    //
    // Class mean:
    //
    // m_y = 1/N_y sum_{j:y_j==y} x_j
    //
    // Class crossnorms:
    //
    // n_yz = m_y'.m_z
    //      = 1/(N_y.N_z) sum_{i,j:y_i==y,y_j==z) x_i'.x_j
    //      = 1/(N_y.N_z) sum_{i,j:y_i==y,y_j==z) K_ij
    // n_yy = n_y
    //
    // Distance to mean:
    //
    // d(m_y,x_k) = sqrt( (m_y-x_k)'.(m_y-x_k) )
    //            = sqrt( x_k'.x_k + m_y'.m_y - 2.m_y.x_k )
    //            = sqrt( x_k'.x_k + n_yy - 2/N_y sum_{j:y_j==y} x_j'.x_k )
    //            = sqrt( K_kk + n_yy - 2/N_y sum_{j:y_j==y} K_jk )
    //
    // Interclass distance:
    //
    // d(m_y,m_z) = sqrt( (m_y-m_z)'.(m_y-m_z) )
    //            = sqrt( m_y'.m_y + m_z'.m_z - 2.m_y'.m_z )
    //            = sqrt( n_yy + n_zz -  2.n_yz )
    //
    // Sphere radius:
    //
    // r_y = max_{i:y_i=y} d(m_y,x_i)

    int N = ml.N();
    int NC = ml.numInternalClasses();
    Vector<Vector<int> > indvec(NC);

    for ( i = 0 ; i < NC ; i++ )
    {
        for ( j = 0 ; j < N ; j++ )
        {
            if ( ml.getInternalClass((ml.y())(j)) == i )
            {
                k = indvec(i).size();

                indvec("&",i).add(k);
                (indvec("&",i))("&",k) = j;
            }
        }
    }

    for ( i = NC-1 ; i >= 0 ; i-- )
    {
        if ( !(indvec.size()) )
        {
            indvec.remove(i);
            NC--;
        }
    }

    Matrix<double> nyz(NC,NC);
    Matrix<double> dyk(NC,N);
    Matrix<double> dyz(NC,NC);
    Vector<double> ry(NC);

    for ( i = 0 ; i < NC ; i++ )
    {
        for ( j = 0 ; j < NC ; j++ )
        {
            nyz("&",i,j) = 0;

            for ( k = 0 ; k < indvec(i).size() ; k++ )
            {
                for ( l = 0 ; l < indvec(j).size() ; l++ )
                {
                    nyz("&",i,j) += calcK(samekern,ml,distkern,(indvec(i))(k),(indvec(j))(l));
                }
            }

            nyz("&",i,j) /= indvec(i).size();
            nyz("&",i,j) /= indvec(j).size();
        }
    }

    for ( i = 0 ; i < NC ; i++ )
    {
        for ( j = 0 ; j < N ; j++ )
        {
            dyk("&",i,j) = 0;

            for ( k = 0 ; k < indvec(i).size() ; k++ )
            {
                dyk("&",i,j) += calcK(samekern,ml,distkern,(indvec(i))(k),j);
            }

            dyk("&",i,j) *= -2.0/((indvec(i)).size());
            dyk("&",i,j) += calcK(samekern,ml,distkern,j,j);
            dyk("&",i,j) += nyz(i,i);

            dyk("&",i,j) = sqrt(dyk("&",i,j));
        }
    }

    for ( i = 0 ; i < NC ; i++ )
    {
        for ( j = 0 ; j < NC ; j++ )
        {
            dyz("&",i,j) = sqrt(nyz(i,i)+nyz(j,j)-(2*nyz(i,j)));
        }
    }

    retVector<double> tmpva;
    retVector<double> tmpvb;

    for ( i = 0 ; i < NC ; i++ )
    {
        ry("&",i) = max(dyk(i,indvec(i),tmpva,tmpvb),j);
    }

    Vector<double> d_l(N);
    Vector<double> d_d(N);
    Vector<double> d(N);
    Vector<double> r_l(N);
    Vector<double> r_d(N);
    Vector<double> g_x(N);

    for ( i = 0 ; i < NC ; i++ )
    {
        for ( j = 0 ; j < indvec(i).size() ; j++ )
        {
            d_l("&",(indvec(i))(j)) = dyk(i,(indvec(i))(j));
            r_l("&",(indvec(i))(j)) = ry(i);

            if ( NC > 1 )
            {
                int noty = -1;
                double mindist = 0.0;

                for ( k = 0 ; k < NC ; k++ )
                {
                    if ( ( k != i ) && ( ( dyz(i,k) < mindist ) || ( noty == -1 ) ) )
                    {
                        noty = k;
                        mindist = dyz(i,k);
                    }
                }

                d_d("&",(indvec(i))(j)) = dyk(noty,(indvec(i))(j));
                d("&",(indvec(i))(j))   = dyz(i,noty);
                r_d("&",(indvec(i))(j)) = ry(noty);
            }
        }
    }

    if ( ((fuzzfn.varsUsed())(2))(6) || ((fuzzfn.varsUsed())(2))(10) )
    {
        svmvolatile int killSwitch = 0;
        
        for ( i = 0 ; i < NC ; i++ )
        {
            SVM_Single clustersvm;

            clustersvm.setKernel(distkern);

            for ( j = 0 ; j < indvec(i).size() ; j++ )
            {
                clustersvm.addTrainingVector(j,ml.x(indvec(i)(j)));
            }

            clustersvm.xferx(ml);

            int dummy = 0; //FIXME: should check return value
            clustersvm.autosetLinBiasForce(nu);
            clustersvm.train(dummy,killSwitch);

            for ( j = 0 ; j < indvec(i).size() ; j++ )
            {
                gentype tempg_x,tempresh;

                //clustersvm.gTrainingVector(g_x("&",(indvec(i))(j)),dummya,j,dummyb);
                clustersvm.ghTrainingVector(tempresh,tempg_x,j);

                g_x("&",(indvec(i))(j)) = (double) tempg_x;
            }
        }
    }

    SparseVector<SparseVector<gentype> > locvar(argvariables);

    (locvar("&",3))("&",0) = f;
    (locvar("&",3))("&",1) = m;
    (locvar("&",3))("&",2) = nu;

    Vector<double> Cweightfuzz(ml.Cweightfuzz());
    Vector<double> epsweight(ml.epsweight());

    for ( i = 0 ; i < NC ; i++ )
    {
        for ( j = 0 ; j < indvec(i).size() ; j++ )
        {
            (locvar("&",2))("&",1)  = d_l((indvec(i))(j));
            (locvar("&",2))("&",2)  = d_d((indvec(i))(j));
            (locvar("&",2))("&",3)  = d((indvec(i))(j));
            (locvar("&",2))("&",4)  = r_l((indvec(i))(j));
            (locvar("&",2))("&",5)  = r_d((indvec(i))(j));
            (locvar("&",2))("&",6)  = g_x((indvec(i))(j));
            (locvar("&",2))("&",7)  = 0.5+((exp(f*(d_d((indvec(i))(j))-d_l((indvec(i))(j)))/d((indvec(i))(j)))-exp(-f))/(2*(exp(f)-exp(-f))));
            (locvar("&",2))("&",8)  = pow(((2*(0.5+((exp(f*(d_d((indvec(i))(j))-d_l((indvec(i))(j)))/d((indvec(i))(j)))-exp(-f))/(2*(exp(f)-exp(-f))))))-1),m);
            (locvar("&",2))("&",9)  = 0.5+((1-(d_l((indvec(i))(j))/(r_l((indvec(i))(j))+f)))/2);
            (locvar("&",2))("&",10) = 0.5*(1+tanh(f*((2*g_x((indvec(i))(j)))+m)));

            if ( setCoreps )                                
            {
                (locvar("&",2))("&",0) = (ml.Cweightfuzz())((indvec(i))(j));
                Cweightfuzz("&",((indvec(i))(j))) = (double) fuzzfn(locvar);
            }

            else
            {
                (locvar("&",2))("&",0)  = (ml.epsweight())((indvec(i))(j));
                epsweight("&",((indvec(i))(j))) = (double) fuzzfn(locvar);
            }
        }
    }

    for ( i = 0 ; i < NC ; i++ )
    {
        for ( j = 0 ; j < indvec(i).size() ; j++ )
        {
            if ( setCoreps )
            {
                ml.setCweightfuzz(((indvec(i))(j)),Cweightfuzz("&",((indvec(i))(j))));
            }

            else
            {
                ml.setepsweight(((indvec(i))(j)),epsweight("&",((indvec(i))(j))));
            }
        }
    }

    return res;
}


double calcK(int samekern, const ML_Base &ml, const MercerKernel &distkern, int i, int j)
{
    double res;

    if ( samekern )
    {
        ml.K2(res,i,j);
    }

    else
    {
        gentype tempres;
        ml.K2(tempres,i,j,distkern);
        res = (double) tempres;
    }

    return res;
}

