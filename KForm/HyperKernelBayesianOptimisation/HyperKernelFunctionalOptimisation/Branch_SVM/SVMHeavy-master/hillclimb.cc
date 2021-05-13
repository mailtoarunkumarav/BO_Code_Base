
//
// Hill-climbing feature selector for SVMs
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "hillclimb.h"
#include "errortest.h"


double optFeatHillClimb(ML_Base &svm, int n, int m, int rndit, Vector<int> &usedfeats, std::ostream &logdest, int useDescent, const Vector<SparseVector<gentype> > &xtest, const Vector<gentype> &ytest, int startpoint, int traverse, int startdirty)
{
    double baselineres = 1e6;
    int i,j,k,l;
    int bj = -1;
    int bk = -1;
    int bl = -1;
    double bestglobalerr = 1e6;
    double bestlocalerr = 1e6;
    double localerr = 1e6;
    int dummyres = 0; //FIXME: should check return value

    Vector<int> cnt;
    Matrix<int> cfm;

    Vector<double> repres;
    Vector<double> dcnt;
    Matrix<double> dcfm;

    usedfeats.resize(0);

    if ( svm.N() )
    {
	// ind_used lists currently used indices (initially empty)
	// ind_unused lists currently unused indices (initially all)

	Vector<int> ind_used;
	Vector<int> ind_unused;
        Vector<int> baseline_ind;

        // Find all available indices

        SparseVector<gentype> xsum;

        svm.xsum(xsum);

        if ( xsum.indsize() )
        {
            for ( i = 0 ; i < xsum.indsize() ; i++ )
            {
                ind_unused.add(i);
                ind_unused("&",i) = xsum.ind(i);
            }
        }

        // Set baseline indices

        if ( !(svm.getKernel_unsafe().isIndex()) )
        {
            if ( xsum.indsize() )
            {
                for ( i = 0 ; i < xsum.indsize() ; i++ )
                {
                    baseline_ind.add(i);
                    baseline_ind("&",i) = xsum.ind(i);
                }
            }
        }

        else
        {
            baseline_ind = svm.getKernel_unsafe().cIndexes();
        }

        // If startdirty then make ind_used and ind_unused match up with
        // what is currently used/unused in the SVM kernel.

        if ( startdirty && svm.getKernel_unsafe().isIndex() && svm.getKernel_unsafe().cIndexes().size() )
        {
            for ( i = 0 ; i < svm.getKernel_unsafe().cIndexes().size() ; i++ )
            {
                if ( ind_unused.size() )
                {
                    for ( j = 0 ; j < ind_unused.size() ; j++ )
                    {
                        if ( ind_unused(j) == (svm.getKernel_unsafe().cIndexes())(i) )
                        {
                            ind_unused.remove(j);
                            ind_used.add(ind_used.size());
                            ind_used("&",ind_used.size()-1) = (svm.getKernel_unsafe().cIndexes())(i);

                            break;
                        }
                    }
                }

                else
                {
                    break;
                }
            }

            if ( useDescent )
            {
                qswap(ind_used,ind_unused);
            }
        }

if ( useDescent )
{
logdest << "# "; // << ind_unused;
}

else
{
logdest << "@ "; // << ind_unused;
}
	// Start the search

        if ( ind_unused.size() )
	{
            // Preliminary training (we do this here just in case hill
            // climbing doesn't actually improve anything)

            //svm.getKernel_unsafe().setIndex();
            //svm.getKernel_unsafe().setIndexes(baseline_ind);
            //svm.resetKernel();
            //svm.train();

            // Record baseline result

            if ( n == -3 )
            {
                baselineres = calcTest(svm,xtest,ytest,cnt,cfm);
            }

            else if ( n == -2 )
            {
                baselineres = calcRecall(svm,cnt,cfm);
            }

            else if ( n == -1 )
            {
                baselineres = calcLOO(svm,cnt,cfm,startpoint);
            }

            else if ( m < 0 )
            {
                baselineres = calcCross(svm,n);
            }

            else
            {
                baselineres = calcCross(svm,n,rndit,repres,dcnt,dcfm,m,startpoint);
            }
logdest << " (~" << baselineres << "," << baseline_ind << "~) ";

            // Set initial indices

            //svm.getKernel_unsafe().setIndexes();
            svm.resetKernel();

	    // Turn on kernel indexing, if not already selected

            if ( !useDescent )
            {
                svm.getKernel_unsafe().setIndexes(ind_used);
                svm.resetKernel();
            }

            else
            {
                svm.getKernel_unsafe().setIndexes(ind_unused);
                svm.resetKernel();
            }

            // Preliminary training

            svm.train(dummyres);

            // Record baseline result

            if ( n == -3 )
            {
                bestglobalerr = calcTest(svm,xtest,ytest,cnt,cfm);
            }

            else if ( n == -2 )
            {
                bestglobalerr = calcRecall(svm,cnt,cfm);
            }

            else if ( n == -1 )
            {
                bestglobalerr = calcLOO(svm,cnt,cfm,startpoint);
            }

            else if ( m < 0 )
            {
                bestglobalerr = calcCross(svm,n);
            }

            else
            {
                bestglobalerr = calcCross(svm,n,rndit,repres,dcnt,dcfm,m,startpoint);
            }
logdest << " (@" << bestglobalerr << "@) ";

            while ( ind_unused.size() )
	    {
		// Add each index, test, save if best

		for ( j = 0 ; j < ind_unused.size() ; j++ )
		{
		    // Grab index from unused, work out where it fits in used

		    k = ind_unused(j);
		    l = 0;

		    if ( ind_used.size() )
		    {
			while ( ( ind_used(l) < k ) && ( l < ind_used.size() ) )
			{
			    l++;

			    if ( l == ind_used.size() )
			    {
                                break;
			    }
			}
		    }

                    if ( !useDescent )
                    {
                        // Temporarily add to used

                        ind_used.add(l);
                        ind_used("&",l) = k;

                        // Temporarily set indices and train

                        svm.getKernel_unsafe().setIndexes(ind_used);
                        svm.resetKernel();
                    }

                    else
                    {
                        // Temporarily remove from unused

                        ind_unused.remove(j);

                        // Temporarily set indices and train

                        svm.getKernel_unsafe().setIndexes(ind_unused);
                        svm.resetKernel();
                    }

                    svm.train(dummyres);

                    // Assess performance

logdest << "* ";
                    if ( n == -3 )
		    {
                        localerr = calcTest(svm,xtest,ytest,cnt,cfm);
                    }

		    else if ( n == -2 )
		    {
                        localerr = calcRecall(svm,cnt,cfm);
		    }

		    else if ( n == -1 )
		    {
                        localerr = calcLOO(svm,cnt,cfm,startpoint);
		    }

		    else if ( m < 0 )
		    {
                        localerr = calcCross(svm,n);
		    }

		    else
		    {
                        localerr = calcCross(svm,n,rndit,repres,dcnt,dcfm,m,startpoint);
		    }

                    // Record details if this is the best choice in this round

		    if ( ( localerr < bestlocalerr ) || !j )
		    {
			bestlocalerr = localerr;

			bj = j;
			bk = k;
			bl = l;
		    }

                    if ( !useDescent )
                    {
                        // Remove from indices used

                        ind_used.remove(l);
                    }

                    else
                    {
                        // Add to indices unused

                        ind_unused.add(j);
                        ind_unused("&",j) = k;
                    }
		}

		// Stop if no improvement, training on the way out

logdest << bestlocalerr << " - \n"; // << ind_used << "\n";
		if ( bestlocalerr > bestglobalerr )
		{
                    if ( !useDescent )
                    {
                        svm.getKernel_unsafe().setIndexes(ind_used);
                        svm.resetKernel();
                    }

                    else
                    {
                        svm.getKernel_unsafe().setIndexes(ind_unused);
                        svm.resetKernel();
                    }

                    svm.train(dummyres);

		    break;
		}

		// Save new best error

		bestglobalerr = bestlocalerr;

		// Add to indices used, remove from unused

		ind_unused.remove(bj);

		ind_used.add(bl);
		ind_used("&",bl) = bk;

                if ( !useDescent )
                {
                    svm.getKernel_unsafe().setIndexes(ind_used);
                    svm.resetKernel();
                }

                else
                {
                    svm.getKernel_unsafe().setIndexes(ind_unused);
                    svm.resetKernel();
                }

                svm.train(dummyres);
	    }
	}

        else
        {
            if ( !useDescent )
            {
                usedfeats = ind_used;
            }

            else
            {
                usedfeats = ind_unused;
            }
        }

        if ( baselineres < bestglobalerr )
        {
            bestglobalerr = baselineres;

            usedfeats = baseline_ind;

            svm.getKernel_unsafe().setIndexes(usedfeats);
            svm.resetKernel();
            svm.train(dummyres);
        }

        if ( traverse > 1 )
        {
            return optFeatHillClimb(svm,n,m,rndit,usedfeats,logdest,!useDescent,xtest,ytest,startpoint,traverse-1,1);
        }
    }

    return bestglobalerr;
}

