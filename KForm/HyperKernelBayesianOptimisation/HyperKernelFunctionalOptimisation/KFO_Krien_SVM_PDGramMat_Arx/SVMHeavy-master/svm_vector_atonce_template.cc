
//
// Vector (at once) regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_vector_atonce_template.h"


evalCacheFn getsigmacallback(const double &dummy)
{
    (void) dummy;

    return &evalSigmaSVM_Vector_atonce_temp_double;
}

evalCacheFn getsigmacallback(const Matrix<double> &dummy)
{
    (void) dummy;

    return &evalSigmaSVM_Vector_atonce_temp_matrix;
}

void evalSigmaSVM_Vector_atonce_temp_double(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    gentype tempres;

    SVM_Vector_atonce_temp<double> *realOwner = (SVM_Vector_atonce_temp<double> *) owner;

    NiceAssert( realOwner );

    tempres = (*(realOwner->GpGrad))(i,i)+(*(realOwner->GpGrad))(j,j)-(2.0*(*(realOwner->GpGrad))(i,j));
    KFinaliser(res,tempres,realOwner->tspaceDim());

    return;
}

void evalSigmaSVM_Vector_atonce_temp_matrix(double &res, int i, int j, const gentype **pxyprod, const void *owner)
{
    (void) pxyprod;

    gentype tempres;

    SVM_Vector_atonce_temp<Matrix<double> > *realOwner = (SVM_Vector_atonce_temp<Matrix<double> > *) owner;

    NiceAssert( realOwner );

    tempres = ((*(realOwner->GpGrad))(i,i))(0,0)+((*(realOwner->GpGrad))(j,j))(0,0)-(2.0*((*(realOwner->GpGrad))(i,j))(0,0));
    KFinaliser(res,tempres,realOwner->tspaceDim());

    return;
}

void KFinaliser(Matrix<double> &res, gentype &src, int tspaceDim)
{
    gentypeToMatrixRep(res,src,tspaceDim);

    return;
}

void KFinaliser(double &res, gentype &src, int tspaceDim)
{
    (void) tspaceDim;

    res = (double) real(src);

    return;
}

void KOffset(Matrix<double> &res, double diagoff, int tspaceDim)
{
    if ( tspaceDim )
    {
        int i;

        for ( i = 0 ; i < tspaceDim ; i++ )
        {
            res("&",i,i) += diagoff;
        }
    }

    return;
}

void KOffset(double &res, double diagoff, int tspaceDim)
{
    (void) tspaceDim;

    res += diagoff;

    return;
}

void redimelmsvm(Vector<double> &x, int olddim, int newdim)
{
    (void) olddim;

    x.resize(newdim);
    return;
}

void addfeatsvm(Vector<double> &x, int iii, int dummy)
{
    (void) dummy;

    x.add(iii);
    return;
}

void removefeatsvm(Vector<double> &x, int iii, int dummy)
{
    (void) dummy;

    x.remove(iii);
    return;
}

int isKreal_nontemp(const double &dummy)
{
    (void) dummy;

    return 1;
}

int isKunreal_nontemp(const double &dummy)
{
    (void) dummy;

    return 0;
}

int isKreal_nontemp(const Matrix<double> &dummy)
{
    (void) dummy;

    return 0;
}

int isKunreal_nontemp(const Matrix<double> &dummy)
{
    (void) dummy;

    return 1;
}

// In visual studio you can't use pointers to template functions, so rather
// than do this all our matrix(cache) allocation in template land we need to
// thunk back here to the world of pointers to non-template functions.  The
// alternative that worked for gcc was much more elegant.
//
// The dummy argument exists purely to make these functions explicitly
// selectable from a template (this probably has a name, design style or
// something).

Matrix<double> *alloc_gp(void *kerncache, int nrows, int ncols, const double &dummy)
{
    (void) dummy;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(Kcache_celm_double,Kcache_crow_double,kerncache,nrows,ncols));

    return res;
}

Matrix<Matrix<double> > *alloc_gp(void *kerncache, int nrows, int ncols, const Matrix<double> &dummy)
{
    (void) dummy;

    Matrix<Matrix<double> > *res;

    MEMNEW(res,Matrix<Matrix<double> >(Kcache_celm_matrix,Kcache_crow_matrix,kerncache,nrows,ncols));

    return res;
}
