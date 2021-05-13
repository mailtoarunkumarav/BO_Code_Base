
//
// Unique ID storage
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "idstore.h"

IDStore::IDStore()
{
    IDmax  = 0;
    refmin = 0;

    return;
}

IDStore::IDStore(const IDStore &src)
{
    *this = src;

    return;
}

IDStore &IDStore::operator=(const IDStore &src)
{
    IDmax   = src.IDmax;
    refmin  = src.refmin;
    IDtoref = src.IDtoref;
    reftoID = src.reftoID;

    return *this;
}

int IDStore::findID(int ref) const
{
    // Recall that the default, setzero value of the ID element
    // of IDelm is -1.  Thus if this accesses an unassigned element of the
    // sparse vector then IDtoref(ref) will be defined by setzero and hence
    // IDtoref(ref).ID will be -1.

    if ( ref < refmin )
    {
	return -1;
    }

    return (int) IDtoref(ref-refmin);
}

int IDStore::findOrAddID(int ref)
{
    int res = findID(ref);

    if ( res == -1 )
    {
	// This bit of code allows for negative references, as may occur
	// if we blindly shove a binary classification problem into
	// a multiclass classifier.

	if ( ref < refmin )
	{
	    IDtoref.offset(refmin-ref);
            refmin = ref;
	}

	reftoID.add(IDmax);
	reftoID("&",IDmax) = ref;
	IDtoref("&",ref-refmin) = IDmax++; // (post-increment is deliberate here)
        res = findID(ref);
    }

    return res;
}

std::ostream &operator<<(std::ostream &output, const IDStore &src )
{
    output << "Number of IDs assigned:     " << src.IDmax   << "\n";
    output << "Minimum reference number:   " << src.refmin  << "\n";
    output << "IDs assigned to references: " << src.IDtoref << "\n";
    output << "References assigned to IDs: " << src.reftoID << "\n";

    return output;
}

std::istream &operator>>(std::istream &input, IDStore &dest)
{
    wait_dummy dummy;

    input >> dummy; input >> dest.IDmax;
    input >> dummy; input >> dest.refmin;
    input >> dummy; input >> dest.IDtoref;
    input >> dummy; input >> dest.reftoID;

    return input;
}
