
//
// Mutable ML pool
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "ml_pool.h"
#include "search.h"
#include <iostream>
#include <sstream>
#include <string>

ML_Pool::ML_Pool() : ML_Mutable()
{
    actMLID = 0; // won't be allocated until requested.
    MLstore.useTightAllocation();

    return;
}

ML_Pool::ML_Pool(const ML_Pool &src) : ML_Mutable()
{
    actMLID = 0; // won't be allocated until requested.
    MLstore.useTightAllocation();

    assign(src,0);

    return;
}

ML_Pool::ML_Pool(const ML_Pool &src, const ML_Base *xsrc) : ML_Mutable()
{
    (void) src;
    (void) xsrc;

    throw("Illegal construct in ML_Pool");

    return;
}

ML_Pool::~ML_Pool()
{
    return;
}

std::ostream &operator<<(std::ostream &output, const ML_Pool &src )
{
    return src.printstream(output);
}

std::istream &operator>>(std::istream &input,        ML_Pool &dest)
{
    return dest.inputstream(input);
}

std::ostream &ML_Pool::printstream(std::ostream &output) const
{
    output << "ML pool\n";

    output << "Active MLID: " << actMLID << "\n";
    output << "ML pool:     " << MLstore << "\n";

    return output;
}

std::istream &ML_Pool::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> actMLID;
    input >> dummy; input >> MLstore;

    return input;
}

int ML_Pool::add(int type)
{
    int MLID = MLstore.size();

    MLstore("&",MLID).setMLTypeClean(type);

    return MLID;
}

void ML_Pool::remove(int MLID)
{
    MLstore.zero(MLID);

    return;
}

void ML_Pool::removeall(void)
{
    MLstore.zero();

    return;
}

void ML_Pool::setactiveML(int MLID)
{
    NiceAssert( MLID >= 0 );

    actMLID = MLID;

    return;
}
