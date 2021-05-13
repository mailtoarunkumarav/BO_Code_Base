
//
// Simple function block
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
#include <fstream>
#include "blk_system.h"


BLK_System::BLK_System(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    return;
}

BLK_System::BLK_System(const BLK_System &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_System::BLK_System(const BLK_System &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_System::~BLK_System()
{
    return;
}

std::ostream &BLK_System::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "User wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_System::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}





































int BLK_System::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    int j;
    int tempresh;

    SparseVector<SparseVector<gentype> > xx;

    xx("&",zeroint()) = x(i);

    if ( getxfilename().length() )
    {
        std::ofstream xfile;

        xfile.open(getxfilename().c_str());

        for ( j = 0 ; j < N() ; j++ )
        {
            printnoparen(xfile,x(j)) << "\n";
        }

        xfile.close();
    }

    if ( getyfilename().length() )
    {
        std::ofstream yfile;

        yfile.open(getyfilename().c_str());

        for ( j = 0 ; j < N() ; j++ )
        {
            yfile << y()(j) << "\n";
        }

        yfile.close();
    }

    if ( getxyfilename().length() )
    {
        std::ofstream xyfile;

        xyfile.open(getxyfilename().c_str());

        for ( j = 0 ; j < N() ; j++ )
        {
            printnoparen(xyfile,x(j)) << "\t" << y()(j) << "\n";
        }

        xyfile.close();
    }

    if ( getyxfilename().length() )
    {
        std::ofstream yxfile;

        yxfile.open(getyxfilename().c_str());

        for ( j = 0 ; j < N() ; j++ )
        {
            yxfile << y()(j) << "\t";
            printnoparen(yxfile,x(j)) << "\n";
        }

        yxfile.close();
    }

//errstream() << "phantomx 0: " << getsyscall() << "\n";
    gentype callfn(getsyscall());

//errstream() << "phantomx 1: " << callfn << "\n";
    callfn = callfn(xx);

//errstream() << "phantomx 2: " << callfn << "\n";
    if ( callfn.isValVector() )
    {
        gentype temp;
        gentype tempsp;

        temp.makeString("");
        tempsp.makeString("");

        for ( j = 0 ; j < callfn.size() ; j++ )
        {
            if ( j )
            {
                temp += tempsp;
//errstream() << "phantomx 3: " << temp << "\n";
            }

            temp += callfn(j);
//errstream() << "phantomx 4: " << temp << "\n";
        }

        callfn = temp;
//errstream() << "phantomx 5: " << callfn << "\n";
    }

    std::string runstring(callfn.cast_string());
//errstream() << "phantomx 6: " << runstring << "\n";
    tempresh = svm_system(runstring.c_str());

    if ( getrfilename().length() )
    {
        std::ifstream rfile;

        rfile.open(getrfilename().c_str());

        rfile >> resg;

        rfile.close();
    }

    else
    {
        resg.force_null();
    }

    resh = resg;
    
    return tempresh;
}

