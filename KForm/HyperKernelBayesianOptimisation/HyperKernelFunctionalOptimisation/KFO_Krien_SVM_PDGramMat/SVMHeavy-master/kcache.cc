
//
// Kernel cache class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "kcache.h"

const Vector<double>  &Kcache_crow_double (int numi, const void *owner)
{
    Kcache<double> *typed_owner = (Kcache<double> *) owner;

    return typed_owner->getrow(numi);
}

const Vector<gentype> &Kcache_crow_gentype(int numi, const void *owner)
{
    Kcache<gentype> *typed_owner = (Kcache<gentype> *) owner;

    return typed_owner->getrow(numi);
}

const Vector<Matrix<double> > &Kcache_crow_matrix(int numi, const void *owner)
{
    Kcache<Matrix<double> > *typed_owner = (Kcache<Matrix<double> > *) owner;

    return typed_owner->getrow(numi);
}

const double  &Kcache_celm_double (int numi, int numj, const void *owner)
{
    Kcache<double> *typed_owner = (Kcache<double> *) owner;

    return typed_owner->getval(numi,numj);
}

const gentype &Kcache_celm_gentype(int numi, int numj, const void *owner)
{
    Kcache<gentype> *typed_owner = (Kcache<gentype> *) owner;

    return typed_owner->getval(numi,numj);
}

const Matrix<double> &Kcache_celm_matrix(int numi, int numj, const void *owner)
{
    Kcache<Matrix<double> > *typed_owner = (Kcache<Matrix<double> > *) owner;

    return typed_owner->getval(numi,numj);
}
