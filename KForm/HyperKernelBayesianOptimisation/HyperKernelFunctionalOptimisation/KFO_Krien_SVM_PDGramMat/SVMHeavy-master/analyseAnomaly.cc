
//
// Anomaly analysis and new class creation
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "analyseAnomaly.h"

int anAnomalyCreate(ML_Mutable &ml, Vector<SparseVector<gentype> > &x, Vector<gentype> &zassign, Vector<int> &trigVect, Vector<int> &anomVect, Vector<int> &addVect, int Nnew, double adist, double nadist, int Mclass, int addVectsToML, int anomalyClass)
{
    NiceAssert( Nnew >= 0 );
    NiceAssert( Mclass >= -1 );
    NiceAssert( adist >= 0.0 );
    NiceAssert( ml.isClassifier() );

    if ( Mclass && !anomalyClass )
    {
        anomalyClass = ml.anomalyClass();
    }

    trigVect.resize(0);
    anomVect.resize(0);
    addVect.resize(0);
    zassign.resize(x.size());

    if ( x.size() )
    {
        gentype aanomalyClass(anomalyClass);
        gentype MMclass(Mclass);

        int i;
        double xdist;

//errstream() << "phantomx: Nnew: " << Nnew << "\n";
//errstream() << "phantomx: adist: " << adist << "\n";
//errstream() << "phantomx: nadist: " << nadist << "\n";
//errstream() << "phantomx: Mclass: " << Mclass << "\n";
//errstream() << "phantomx: addVectsToML: " << addVectsToML << "\n";
//errstream() << "phantomx: anomalyClass: " << anomalyClass << "\n";
errstream() << "Testing training vectors (" << adist << "," << nadist << " - " << anomalyClass << ")\n";
        for ( i = 0 ; i < x.size() ; i++ )
        {
            gentype resh;
            gentype resg;

            ml.gh(resh,resg,x(i),1);

            xdist = (double) resg(ml.findID((int) resh));

//errstream() << "phantomx 0: " << xdist << "," << resh << " (" << ml.findID((int) resh) << ")," << resg;
            if ( Mclass && anomalyClass && ( (int) resh == anomalyClass ) )
            {
                if ( xdist >= adist )
                {
//errstream() << "   !";
                    trigVect.add(trigVect.size());
                    trigVect("&",trigVect.size()-1) = i;

                    zassign("&",i) = Mclass;
                }

                else
                {
//errstream() << "   @";
                    anomVect.add(anomVect.size());
                    anomVect("&",anomVect.size()-1) = i;

                    zassign("&",i) = anomalyClass;
                }
            }

            else
            {
                if ( xdist >= nadist )
                {
//errstream() << "   #";
                    addVect.add(addVect.size());
                    addVect("&",addVect.size()-1) = i;

                    zassign("&",i) = resh;
                }

                else
                {
//errstream() << "   ?";
                    anomVect.add(anomVect.size());
                    anomVect("&",anomVect.size()-1) = i;

                    zassign("&",i) = anomalyClass;
                }
            }
//errstream() << "\n";
        }

errstream() << "Testing triggers (" << trigVect.size() << " possibles)\n";
        if ( ( trigVect.size() >= Nnew ) && Nnew )
        {
            // Anomaly trigger met, add new class to ml and incorporate
            // trigVect into addVect (as they will be added in this case)

            retVector<gentype> tmpva;

            zassign("&",trigVect,tmpva) = MMclass;
            addVect.append(addVect.size(),trigVect);
            ml.addclass(Mclass);
        }

        else
        {
            // Anomaly trigger not met, label all anomalies as anomalous and
            // reset trigVect.

            retVector<gentype> tmpva;

            zassign("&",trigVect,tmpva) = aanomalyClass;
            anomVect.append(anomVect.size(),trigVect);
            trigVect.resize(0);
        }

//errstream() << "Trigger vector: " << trigVect << "\n";
//errstream() << "Anomaly vector: " << anomVect << "\n";
//errstream() << "Addition vector: " << addVect << "\n";
errstream() << "Adding training vectors:\n";
errstream() << addVect << "\n";
        if ( addVectsToML && addVect.size() )
        {
            // Add training vectors as assigned, including new anomaly class if
            // relevant.  Do not label vectors that are anomalous.

            for ( i = 0 ; i < addVect.size() ; i++ )
            {
                if ( ( ( addVectsToML & 1 ) && ( zassign(addVect(i)) != MMclass ) ) ||
                     ( ( addVectsToML & 2 ) && ( zassign(addVect(i)) == MMclass ) )    )
                {
                    ml.qaddTrainingVector(ml.N(),zassign(addVect(i)),x("&",addVect(i)));
                }
            }
        }
    }

    return trigVect.size();
}
