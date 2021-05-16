
//
// Bernstain polynomial block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_bernst.h"


BLK_Bernst::BLK_Bernst(int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    localygood = 0;

    locsampleMode = 0;
    locNsamp      = -1;

    return;
}

BLK_Bernst::BLK_Bernst(const BLK_Bernst &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_Bernst::BLK_Bernst(const BLK_Bernst &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_Bernst::~BLK_Bernst()
{
    return;
}

std::ostream &BLK_Bernst::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Bernstein polynomial block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Bernst::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}

const Vector<gentype> &BLK_Bernst::y(void) const
{
    Vector<gentype> &res = (**thisthisthis).localy;

    if ( !localygood )
    {
        int jj,kk;

        int allowGridSample = 1;
        int dim = locxmin.size();
        int totsamp = allowGridSample ? ( dim ? (int) pow(locNsamp,dim) : 0 ) : locNsamp;

        NiceAssert( locxmax.size() == locxmin.size() );

        gentype xxmin(locxmin);
        gentype xxmax(locxmax);

        res.resize(totsamp);

//errstream() << "phantomxyzq 0: " << locNsamp << "\n";
//errstream() << "phantomxyzq 1: " << locxmin << "\n";
//errstream() << "phantomxyzq 2: " << locxmax << "\n";
        if ( allowGridSample && ( dim == 1 ) )
        {
//errstream() << "phantomxyzq 3\n";
            SparseVector<gentype> xq;

            for ( jj = 0 ; jj < locNsamp ; jj++ )
            {
                xq("&",zeroint())  = (((double) jj)+0.5)/(((double) locNsamp));
                xq("&",zeroint()) *= (locxmax(zeroint())-locxmin(zeroint()));
                xq("&",zeroint()) += locxmin(zeroint());

                gg(res("&",jj),xq);
            }
//errstream() << "phantomxyzq 4: " << xq << "\n";
//errstream() << "phantomxyzq 5: " << res << "\n";
        }

        else if ( allowGridSample )
        {
//errstream() << "phantomxyzq 6\n";
            Vector<int> jjj(dim);

            jjj = zeroint();

            for ( jj = 0 ; jj < totsamp ; jj++ )
            {
                SparseVector<gentype> xq;

                for ( kk = 0 ; kk < dim ; kk++ )
                {
                    xq("&",kk)  = (((double) jjj(kk))+0.5)/(((double) locNsamp));
                    xq("&",kk) *= (locxmax(kk)-locxmin(kk));
                    xq("&",kk) += locxmin(kk);
                }

                gg(res("&",jj),xq);
//errstream() << "phantomxyzq 7: " << xq << "\n";

                for ( kk = 0 ; kk < dim ; kk++ )
                {
                    jjj("&",kk)++;

                    if ( jjj(kk) >= locNsamp )
                    {
                        jjj("&",kk) = 0;
                    }

                    else
                    {
                        break;
                    }
                }
            }
//errstream() << "phantomxyzq 8: " << res << "\n";
        }

        else
        {
            for ( jj = 0 ; jj < totsamp ; jj++ )
            {
                gentype xx = urand(xxmin,xxmax);
                SparseVector<gentype> xq(xx.cast_vector());

                gg(res("&",jj),xq);
            }
//errstream() << "phantomxyzq 9: " << res << "\n";
        }

        (**thisthisthis).localygood = 1;
//errstream() << "phantomxy 0: " << Nmax << "," << locNsamp << "," << localy.size();
//if ( res.size() != 100 ) 
//{ 
//errstream() << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"; 
//for ( ii = 0 ; ii < numReps() ; ii++ )
//{
//errstream() << "phantomxyz " << ii << ": " << getRepConst(ii) << "\n";
//}
//}
//errstream() << "\n";
////errstream() << "phantomxyz localy = " << res << "\n";
    }

    return res;
}

int BLK_Bernst::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    const gentype &n = bernDegree();
    const gentype &j = bernIndex();

    int res = 0;

    const SparseVector<gentype> &xx = x(i);

//errstream() << "phantomxyzqw 0\n";
    if ( xx.size() == 0 )
    {
        NiceAssert( n.isValNull() );
        NiceAssert( j.isValNull() );

        // By default we work on the assumption that the empty product = 1

        resg = 1.0;
        resh = resg;
//errstream() << "phantomxyzqw 1\n";
    }

    else if ( xx.size() == 1 )
    {
//errstream() << "phantomxyzqw 2\n";
        // "Standard" Bernstein basis polynomials.

        NiceAssert( n.isCastableToIntegerWithoutLoss() );
        NiceAssert( j.isCastableToIntegerWithoutLoss() );

        int nn = (int) n;
        int jj = (int) j;

//errstream() << "phantomxyzqw 3: " << nn << "\n";
//errstream() << "phantomxyzqw 4: " << jj << "\n";
        NiceAssert( nn >= 0 );
        NiceAssert( jj <= nn );

        gentype nnn(nn);
        gentype jjj(jj);

//errstream() << "phantomxyzqw 5: " << nnn << "\n";
//errstream() << "phantomxyzqw 6: " << jjj << "\n";
        const gentype &xxx = xx.direcref(0);
        const gentype ov(1.0);

        resg = (pow(xxx,jjj)*pow(ov-xxx,nnn-jjj)*((double) xnCr(nn,jj)));
//errstream() << "phantomxyzqw 7: pow(" << xxx << "," << jjj << " = " << pow(xxx,jjj) << "\n";
//errstream() << "phantomxyzqw 8: pow(" << ov-xxx << "," << nnn-jjj << ") = " << pow(ov-xxx,nnn-jjj) << "\n";
//errstream() << "phantomxyzqw 9: xnCr(" << nn << "," << jj << ") = " << xnCr(nn,jj) << "\n";
//errstream() << "phantomxyzqw 10: " << resg << "\n";
        resh = resg;
    }

    else
    {
        // See http://www.iue.tuwien.ac.at/phd/heitzinger/node17.html

        NiceAssert( n.isValVector() );
        NiceAssert( j.isValVector() );

        NiceAssert( n.size() == xx.size() );
        NiceAssert( j.size() == xx.size() );

        const Vector<gentype> &nn = (const Vector<gentype> &) n;
        const Vector<gentype> &jj = (const Vector<gentype> &) j;

        resg = 1.0;

        int i;

        for ( i = 0 ; i < xx.size() ; i++ )
        {
            const gentype &nnn = nn(i);
            const gentype &jjj = jj(i);

            NiceAssert( (int) nnn >= 0 );
            NiceAssert( (int) jjj <= (int) nnn );

            const gentype &xxx = xx(i);
            const gentype ov(1.0);

            resg *= (pow(xxx,jjj)*pow(ov-xxx,nnn-jjj)*((double) xnCr((int) nnn,(int) jjj)));
        }

        resh = resg;
    }

    return res;
}


