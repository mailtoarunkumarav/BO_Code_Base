
//
// Unique ID storage
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _idstore_h
#define _idstore_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "nonzeroint.h"
#include "sparsevector.h"


class IDStore;

std::ostream &operator<<(std::ostream &output, const IDStore &src );
std::istream &operator>>(std::istream &input,        IDStore &dest);

// Swap function

inline void qswap(IDStore &a, IDStore &b);

class IDStore
{
    friend std::ostream &operator<<(std::ostream &output, const IDStore &src );
    friend std::istream &operator>>(std::istream &input,        IDStore &dest);

    friend inline void qswap(IDStore &a, IDStore &b);

public:

    // Constructors and assignment operators

    IDStore();
    IDStore(const IDStore &src);
    IDStore &operator=(const IDStore &src);

    // Management:
    //
    // findID(ref) finds the ID number assigned to reference ref, return -1 if unassigned
    // findOrAddID(ref) does likewise, but will assign a new ID to reference ref first if unassigned.
    // findref(int ID) find the reference number associated with ID
    //
    // IDs are assigned sequentially starting from 0

    int findID(int ref) const;
    int findOrAddID(int ref);
    int findref(int ID) const { return reftoID(ID); }
    const Vector<int> &getreftoID(void) const { return reftoID; }
    int size(void) const { return IDmax; }

private:

    int IDmax;
    int refmin;
    SparseVector<nzint> IDtoref;
    Vector<int> reftoID;
};

inline void qswap(IDStore &a, IDStore &b)
{
    qswap(a.IDmax  ,b.IDmax  );
    qswap(a.refmin ,b.refmin );
    qswap(a.IDtoref,b.IDtoref);
    qswap(a.reftoID,b.reftoID);

    return;
}

#endif
