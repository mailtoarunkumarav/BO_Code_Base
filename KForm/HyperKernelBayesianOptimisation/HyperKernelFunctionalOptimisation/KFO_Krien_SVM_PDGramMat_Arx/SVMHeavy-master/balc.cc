
//
// Class balancing for ML
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "balc.h"


void balc(ML_Base &svm)
{
    if ( svm.isClassifier() )
    {
        int N = svm.N();
        int m = svm.numClasses();

        if ( ( m > 1 ) && ( N > 1 ) )
        {
            int i,Ni;
            const Vector<int> &d = svm.ClassLabels();

            for ( i = 0 ; i < m ; i++ )
            {
                Ni = svm.NNC(i);

                if ( Ni )
                {
                    svm.setCclass(d(i),((double) N)/((double) Ni));
                }
            }
        }
    }

    return;
}
