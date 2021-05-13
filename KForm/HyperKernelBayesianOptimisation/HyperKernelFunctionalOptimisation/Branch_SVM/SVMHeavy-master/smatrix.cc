
//
// Special matrix class
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "smatrix.h"


const double &fnConstElm(int i, int j, const void *args);
const double &fnValsElm(int i, int j, const void *args);
const double &fnDiagElm(int i, int j, const void *args);
const double &fnOuterElm(int i, int j, const void *args);
const double &fnReflElm(int i, int j, const void *args);
const double &fnRotatElm(int i, int j, const void *args);
const double &fnCutMElm(int i, int j, const void *args);
//const double &fnBlockDiagElm(int i, int j, const void *args);
const double &fnStackElm(int i, int j, const void *args);
const Matrix<double> &fnStackElmMat(int i, int j, const void *args);

const Vector<double> &fnConstRow(int i, const void *args);
const Vector<double> &fnValsRow(int i, const void *args);
const Vector<double> &fnDiagRow(int i, const void *args);
const Vector<double> &fnOuterRow(int i, const void *args);
const Vector<double> &fnReflRow(int i, const void *args);
const Vector<double> &fnRotatRow(int i, const void *args);
const Vector<double> &fnCutMRow(int i, const void *args);
//const Vector<double> &fnBlockDiagRow(int i, const void *args);
const Vector<double> &fnStackRow(int i, const void *args);
const Vector<Matrix<double> > &fnStackRowMat(int i, const void *args);

void fnConstDel(const void *args, void *vargs);
void fnValsDel(const void *args, void *vargs);
void fnDiagDel(const void *args, void *vargs);
void fnOuterDel(const void *args, void *vargs);
void fnReflDel(const void *args, void *vargs);
void fnRotatDel(const void *args, void *vargs);
void fnCutMDel(const void *args, void *vargs);

void fnDiagDelP(const void *args, void *vargs);
void fnOuterDelP(const void *args, void *vargs);
void fnReflDelP(const void *args, void *vargs);
void fnRotatDelP(const void *args, void *vargs);
void fnCutMDelP(const void *args, void *vargs);
//void fnBlockDiagDel(const void *args, void *vargs);
void fnStackDel(const void *args, void *vargs);
void fnStackDelMat(const void *args, void *vargs);

// That's me, over there in the mirror

// -----------------------------------------------------------------------------------------------

Matrix<double> *smIdent(int m, int n)
{
    return smConst(m,n,1.0);
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smConst(int m, int n, double val, double offdiagval)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,2);

    double          *dvals;
    Vector<double>  *rowshow;
    double         **dp;

    MEMNEWARRAY(dvals,double,3);
    MEMNEW(rowshow,Vector<double>(n));
    MEMNEW(dp,double *);

    dvals[0] = offdiagval;
    dvals[1] = val;

    *dp = &(dvals[1]);

    tempargs[0] = (void *) dvals;
    tempargs[1] = (void *) rowshow;
    tempargs[2] = (void *) dp;

    void *args = (void *) tempargs;

    // The above serves no purpose here.  However by redirecting references to
    // val via the third argument we allow both forms of smConst to have a
    // consistent base.  In the alternative form a pointer to val is given,
    // and may be adjusted externally.

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnConstElm,fnConstRow,args,m,n,fnConstDel,args));

    return res;
}

void fnConstDel(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    void **tempargs = (void **) vargs;

    double          *dvals   = (double *)         tempargs[0];
    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];
    double         **dp      = (double **)        tempargs[2];

    MEMDELARRAY(dvals);
    MEMDEL(rowshow);
    MEMDEL(dp);

    MEMDELARRAY(tempargs);

    return;
}

const double &fnConstElm(int i, int j, const void *args)
{
    void **tempargs = (void **) args;

    double  *dvals = (double *)  tempargs[0];
    double **dp    = (double **) tempargs[2]; // Use reference for compatibility with both variants

    return ( i == j ) ? **dp : dvals[0];
}

const Vector<double> &fnConstRow(int i, const void *args)
{
    void **tempargs = (void **) args;

    double  *dvals = (double *)  tempargs[0];
    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];
    double         **dp      = (double **)        tempargs[2];

    (*rowshow)        = dvals[0];
    (*rowshow)("&",i) = **dp;

    return *rowshow;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smConst(int m, int n, double *val)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,3);

    double          *dvals;
    Vector<double>  *rowshow;
    double         **dp;

    MEMNEWARRAY(dvals,double,1);
    MEMNEW(rowshow,Vector<double>(n));
    MEMNEW(dp,double *);

    dvals[0] = 0.0;

    *dp = val;

    tempargs[0] = (void *) dvals;
    tempargs[1] = (void *) rowshow;
    tempargs[2] = (void *) dp;

    void *args = (void *) tempargs;

    // c/f previous incantation.

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnConstElm,fnConstRow,args,m,n,fnConstDel,args));

    return res;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smOnes(int m, int n)
{
    return smVals(m,n,1.0);
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smZeros(int m, int n)
{
    return smVals(m,n,0.0);
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smVals(int m, int n, double val)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,2);

    double          *dvals;
    Vector<double>  *rowshow;

    MEMNEWARRAY(dvals,double,1);
    MEMNEW(rowshow,Vector<double>(n));

    dvals[0] = val;

    *rowshow = val;

    tempargs[0] = (void *) dvals;
    tempargs[1] = (void *) rowshow;

    void *args = (void *) tempargs;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnValsElm,fnValsRow,args,m,n,fnValsDel,args));

    return res;
}

void fnValsDel(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    void **tempargs = (void **) vargs;

    double          *dvals   = (double *)         tempargs[0];
    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];

    MEMDELARRAY(dvals);
    MEMDEL(rowshow);

    MEMDELARRAY(tempargs);

    return;
}

const double &fnValsElm(int i, int j, const void *args)
{
    (void) i;
    (void) j;

    void **tempargs = (void **) args;

    double *dvals = (double *) tempargs[0];

    return dvals[0];
}

const Vector<double> &fnValsRow(int i, const void *args)
{
    (void) i;

    void **tempargs = (void **) args;

    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];

    return *rowshow;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smDiag(const Vector<double> &b)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,3);

    double         *dvals;
    Vector<double> *rowshow;
    Vector<double> *rowget;

    MEMNEWARRAY(dvals,double,2);
    MEMNEW(rowshow,Vector<double>(b));
    MEMNEW(rowget,Vector<double>(b));

    dvals[0] = 0.0;
    dvals[1] = 0.0;

    tempargs[0] = (void *) dvals;
    tempargs[1] = (void *) rowshow;
    tempargs[2] = (void *) rowget;

    void *args = (void *) tempargs;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnDiagElm,fnDiagRow,args,b.size(),b.size(),fnDiagDel,args));

    return res;
}

void fnDiagDel(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    void **tempargs = (void **) vargs;

    double          *dvals   = (double *)         tempargs[0];
    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];
    Vector<double>  *rowget  = (Vector<double> *) tempargs[2];

    MEMDELARRAY(dvals);
    MEMDEL(rowshow);
    MEMDEL(rowget);

    MEMDELARRAY((void **) vargs);

    return;
}

const double &fnDiagElm(int i, int j, const void *args)
{
    void **tempargs = (void **) args;

    double          *dvals   = (double *)         tempargs[0];
    Vector<double>  *rowget  = (Vector<double> *) tempargs[2];

    return ( dvals[0] = ( ( i == j ) ? (*rowget)(i) : dvals[1] ) );
}

const Vector<double> &fnDiagRow(int i, const void *args)
{
    void **tempargs = (void **) args;

    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];
    Vector<double>  *rowget  = (Vector<double> *) tempargs[2];

    (*rowshow)        = 0.0;
    (*rowshow)("&",i) = (*rowget)(i);

    return *rowshow;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smDiag(Vector<double> *b)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,3);

    double         *dvals;
    Vector<double> *rowshow;

    MEMNEWARRAY(dvals,double,2);
    MEMNEW(rowshow,Vector<double>(*b));

    dvals[0] = 0.0;
    dvals[1] = 0.0;

    tempargs[0] = (void *) dvals;
    tempargs[1] = (void *) rowshow;
    tempargs[2] = (void *) b;

    void *args = (void *) tempargs;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnDiagElm,fnDiagRow,args,(*b).size(),(*b).size(),fnDiagDelP,args));

    return res;
}

void fnDiagDelP(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    void **tempargs = (void **) vargs;

    double          *dvals   = (double *)         tempargs[0];
    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];

    MEMDELARRAY(dvals);
    MEMDEL(rowshow);

    MEMDELARRAY((void **) vargs);

    return;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smOuter(double a, const Vector<double> &b, const Vector<double> &c)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,5);

    double          *dvals;
    Vector<double>  *rowshow;
    Vector<double>  *vert;
    Vector<double>  *horiz;
    double         **br;

    MEMNEWARRAY(dvals,double,2);
    MEMNEW(rowshow,Vector<double>(c));
    MEMNEW(vert,Vector<double>(b));
    MEMNEW(horiz,Vector<double>(c));
    MEMNEW(br,double *);

    dvals[0] = 0.0;
    dvals[1] = a;

    *br = &(dvals[1]);

    tempargs[0] = (void *) dvals;
    tempargs[1] = (void *) rowshow;
    tempargs[2] = (void *) vert;
    tempargs[3] = (void *) horiz;
    tempargs[4] = (void *) br;

    void *args = (void *) tempargs;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnOuterElm,fnOuterRow,args,b.size(),c.size(),fnOuterDel,args));

    return res;
}

void fnOuterDel(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    void **tempargs = (void **) vargs;

    double          *dvals   = (double *)         tempargs[0];
    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];
    Vector<double>  *vert    = (Vector<double> *) tempargs[2];
    Vector<double>  *horiz   = (Vector<double> *) tempargs[3];
    double         **br      = (double **)        tempargs[4];

    MEMDELARRAY(dvals);
    MEMDEL(rowshow);
    MEMDEL(vert);
    MEMDEL(horiz);
    MEMDEL(br);

    MEMDELARRAY((void **) vargs);

    return;
}

const double &fnOuterElm(int i, int j, const void *args)
{
    NiceAssert(args);

    void **tempargs = (void **) args;

    double          *dvals   = (double *)         tempargs[0];
    Vector<double>  *vert    = (Vector<double> *) tempargs[2];
    Vector<double>  *horiz   = (Vector<double> *) tempargs[3];
    double         **br      = (double **)        tempargs[4];

    return ( dvals[0] = ( (**br) * (*vert)(i) * (*horiz)(j) ) );
}

const Vector<double> &fnOuterRow(int i, const void *args)
{
    NiceAssert(args);

    void **tempargs = (void **) args;

    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];
    Vector<double>  *vert    = (Vector<double> *) tempargs[2];
    Vector<double>  *horiz   = (Vector<double> *) tempargs[3];
    double         **br      = (double **)        tempargs[4];

    *rowshow  = (*horiz);
    *rowshow *= (*vert)(i);
    *rowshow *= (**br);

    return *rowshow;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smOuter(double *a, Vector<double> *b, Vector<double> *c)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,5);

    double          *dvals;
    Vector<double>  *rowshow;
    double         **br;

    MEMNEWARRAY(dvals,double,2);
    MEMNEW(rowshow,Vector<double>(*c));
    MEMNEW(br,double *);

    dvals[0] = 0.0;
    dvals[1] = *a;

    *br = &(dvals[1]);

    tempargs[0] = (void *) dvals;
    tempargs[1] = (void *) rowshow;
    tempargs[2] = (void *) b;
    tempargs[3] = (void *) c;
    tempargs[4] = (void *) br;

    void *args = (void *) tempargs;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnOuterElm,fnOuterRow,args,(*b).size(),(*c).size(),fnOuterDelP,args));

    return res;
}

void fnOuterDelP(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    void **tempargs = (void **) vargs;

    double          *dvals   = (double *)         tempargs[0];
    Vector<double>  *rowshow = (Vector<double> *) tempargs[1];
    double         **br      = (double **)        tempargs[4];

    MEMDELARRAY(dvals);
    MEMDEL(rowshow);
    MEMDEL(br);

    MEMDELARRAY((void **) vargs);

    return;
}



// -----------------------------------------------------------------------------------------------

//phantomx

Matrix<double> *smRefl(double d, double a, const Vector<double> &b, const Vector<double> &c)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,5);

    void *args = (void *) tempargs;

    MEMNEWVOIDARRAY(((void **) args)[0],double,3);
    MEMNEWVOID(     ((void **) args)[1],Vector<double>(c));
    MEMNEWVOID(     ((void **) args)[2],Vector<double>(b));
    MEMNEWVOID(     ((void **) args)[3],Vector<double>(c));
    MEMNEWVOIDARRAY(((void **) args)[4],double *,2);

    ((double *) (((void **) args)[0]))[0] = 0.0;
    ((double *) (((void **) args)[0]))[1] = a;
    ((double *) (((void **) args)[0]))[2] = d;
    ((double **) (((void **) args)[4]))[0] = &((double *) (((void **) args)[0]))[1];
    ((double **) (((void **) args)[4]))[1] = &((double *) (((void **) args)[0]))[2];

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnReflElm,fnReflRow,args,b.size(),c.size(),fnReflDel,args));

    return res;
}

const double &fnReflElm(int i, int j, const void *args)
{
    return ( ((double *) (((void **) args)[0]))[0] = ( (*(((double **) (((void **) args)[4]))[0]))*((*((Vector<double> *) (((void **) args)[2])))(i))*((*((Vector<double> *) (((void **) args)[3])))(j)) ) + ( ( i == j ) ? (*(((double **) (((void **) args)[4]))[1])) : 0.0 ) );
}

const Vector<double> &fnReflRow(int i, const void *args)
{
    (*((Vector<double> *) (((void **) args)[1])))  = (*((Vector<double> *) (((void **) args)[3])));
    (*((Vector<double> *) (((void **) args)[1]))) *= (*((Vector<double> *) (((void **) args)[2])))(i);
    (*((Vector<double> *) (((void **) args)[1]))) *= (*((double **) (((void **) args)[4]))[0]);

    (*((Vector<double> *) (((void **) args)[1])))("&",i) += (*((double **) (((void **) args)[4]))[1]);

    return (*((Vector<double> *) (((void **) args)[1])));
    i = 0;
}

void fnReflDel(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    MEMDELVOID((double *) ((void **) vargs)[0]);
    MEMDELVOID((Vector<double> *) ((void **) vargs)[1]);
    MEMDELVOID((Vector<double> *) ((void **) vargs)[2]);
    MEMDELVOID((Vector<double> *) ((void **) vargs)[3]);
    MEMDELVOID((double **) ((void **) vargs)[4]);
    MEMDELARRAY((void **) vargs);

    return;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smRefl(double *d, double *a, Vector<double> *b, Vector<double> *c)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,5);

    void *args = (void *) tempargs;

    MEMNEWVOIDARRAY(((void **) args)[0],double,1);
    MEMNEWVOID(     ((void **) args)[1],Vector<double>(*c));
    ((void **) args)[2] = (void *) b;
    ((void **) args)[3] = (void *) c;
    MEMNEWVOIDARRAY(((void **) args)[4],double *,2);

    ((double *) (((void **) args)[0]))[0] = 0.0;
    ((double **) (((void **) args)[4]))[0] = a;
    ((double **) (((void **) args)[4]))[1] = d;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnReflElm,fnReflRow,args,b->size(),c->size(),fnReflDelP,args));

    return res;
}

void fnReflDelP(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    MEMDELVOID((double *) ((void **) vargs)[0]);
    MEMDELVOID((Vector<double> *) ((void **) vargs)[1]);
    //MEMDELVOID((Vector<double> *) ((void **) vargs)[2]);
    //MEMDELVOID((Vector<double> *) ((void **) vargs)[3]);
    MEMDELVOID((double **) ((void **) vargs)[4]);
    MEMDELARRAY((void **) vargs);

    return;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smRotat(double sina, double cosa, const Vector<double> &b)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,4);

    void *args = (void *) tempargs;

    MEMNEWVOIDARRAY(((void **) args)[0],double,3);
    MEMNEWVOID(     ((void **) args)[1],Vector<double>((b.size())+1));
    MEMNEWVOID(     ((void **) args)[2],Vector<double>(b));
    MEMNEWVOIDARRAY(((void **) args)[3],double *,2);

    ((double *) (((void **) args)[0]))[0] = 0.0;
    ((double *) (((void **) args)[0]))[1] = sina;
    ((double *) (((void **) args)[0]))[2] = cosa;
    ((double **) (((void **) args)[3]))[0] = &((double *) (((void **) args)[0]))[1];
    ((double **) (((void **) args)[3]))[1] = &((double *) (((void **) args)[0]))[2];

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnRotatElm,fnRotatRow,args,(b.size())+1,(b.size())+1,fnRotatDel,args));

    return res;
}

const double &fnRotatElm(int i, int j, const void *args)
{
    if ( i && j )
    {
	((double *) (((void **) args)[0]))[0] =  (*((Vector<double> *) (((void **) args)[2])))(i-1)*(*((Vector<double> *) (((void **) args)[2])))(j-1)*(*(((double **) (((void **) args)[3]))[1]));
    }

    else if ( i )
    {
	((double *) (((void **) args)[0]))[0] = -(*((Vector<double> *) (((void **) args)[2])))(i-1)*(*(((double **) (((void **) args)[3]))[0]));
    }

    else if ( j )
    {
	((double *) (((void **) args)[0]))[0] =  (*((Vector<double> *) (((void **) args)[2])))(j-1)*(*(((double **) (((void **) args)[3]))[0]));
    }

    else
    {
	((double *) (((void **) args)[0]))[0] = (*(((double **) (((void **) args)[3]))[1]));
    }

    return ((double *) (((void **) args)[0]))[0];
}

const Vector<double> &fnRotatRow(int i, const void *args)
{
    (void) i;

    retVector<double> tmpva;

    if ( i )
    {
	(*((Vector<double> *) (((void **) args)[1])))("&",0) = -(*(((double **) (((void **) args)[3]))[0]));
	(*((Vector<double> *) (((void **) args)[1])))("&",1,1,((*((Vector<double> *) (((void **) args)[1]))).size())-1,tmpva)  = (*((Vector<double> *) (((void **) args)[2])));
	(*((Vector<double> *) (((void **) args)[1])))("&",1,1,((*((Vector<double> *) (((void **) args)[1]))).size())-1,tmpva) *= (*(((double **) (((void **) args)[3]))[1]));
        (*((Vector<double> *) (((void **) args)[1]))) *= (*((Vector<double> *) (((void **) args)[2])))(i-1);
    }

    else
    {
	(*((Vector<double> *) (((void **) args)[1])))("&",0) = (*(((double **) (((void **) args)[3]))[1]));
	(*((Vector<double> *) (((void **) args)[1])))("&",1,1,((*((Vector<double> *) (((void **) args)[1]))).size())-1,tmpva)  = (*((Vector<double> *) (((void **) args)[2])));
	(*((Vector<double> *) (((void **) args)[1])))("&",1,1,((*((Vector<double> *) (((void **) args)[1]))).size())-1,tmpva) *= (*(((double **) (((void **) args)[3]))[0]));
    }

    return (*((Vector<double> *) (((void **) args)[1])));
}

void fnRotatDel(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    MEMDELVOID((double *) ((void **) vargs)[0]);
    MEMDELVOID((Vector<double> *) ((void **) vargs)[1]);
    MEMDELVOID((Vector<double> *) ((void **) vargs)[2]);
    MEMDELVOID((double **) ((void **) vargs)[3]);
    MEMDELARRAY((void **) vargs);

    return;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smRotat(double *sina, double *cosa, Vector<double> *b)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,4);

    void *args = (void *) tempargs;

    MEMNEWVOIDARRAY(((void **) args)[0],double,1);
    MEMNEWVOID(     ((void **) args)[1],Vector<double>((b->size())+1));
    ((void **) args)[2] = (void *) b;
    MEMNEWVOIDARRAY(((void **) args)[3],double *,2);

    ((double *) (((void **) args)[0]))[0] = 0.0;
    ((double **) (((void **) args)[3]))[0] = sina;
    ((double **) (((void **) args)[3]))[1] = cosa;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnRotatElm,fnRotatRow,args,(b->size())+1,(b->size())+1,fnRotatDelP,args));

    return res;
}

void fnRotatDelP(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    MEMDELVOID((double *) ((void **) vargs)[0]);
    MEMDELVOID((Vector<double> *) ((void **) vargs)[1]);
    //MEMDELVOID((Vector<double> *) ((void **) vargs)[2]);
    MEMDELVOID((double **) ((void **) vargs)[3]);
    MEMDELARRAY((void **) vargs);

    return;
}



// -----------------------------------------------------------------------------------------------

Matrix<double> *smCutM(int m, int s, int t)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,5);

    void *args = (void *) tempargs;

    m--;

    MEMNEWVOIDARRAY(((void **) args)[0],double,1);
    MEMNEWVOIDARRAY(((void **) args)[1],int,3);
    MEMNEWVOID(     ((void **) args)[2],Vector<double>( ( t == -1 ) ? m+1 : m ));
    MEMNEWVOIDARRAY(((void **) args)[3],int *,2);

    ((double *) (((void **) args)[0]))[0] = 0.0;
    ((int *) (((void **) args)[1]))[0] = m;
    ((int *) (((void **) args)[1]))[1] = s;
    ((int *) (((void **) args)[1]))[2] = t;
    ((int **) (((void **) args)[3]))[0] = &((int *) (((void **) args)[1]))[1];
    ((int **) (((void **) args)[3]))[1] = &((int *) (((void **) args)[1]))[2];

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnCutMElm,fnCutMRow,args,( ( s == -1 ) ? m+1 : m ),( ( t == -1 ) ? m+1 : m ),fnCutMDel,args));

    ((void **) args)[4] = (void *) res;

    return res;
}

const double &fnCutMElm(int i, int j, const void *args)
{
    if ( ( (*(((int **) (((void **) args)[3]))[0])) == -1 ) && ( (*(((int **) (((void **) args)[3]))[1])) == -1 ) )
    {
	if ( ( (*((Matrix<double> *) (((void **) args)[4]))).numRows() != ((int *) (((void **) args)[1]))[0]+1 ) || ( (*((Matrix<double> *) (((void **) args)[4]))).numCols() != ((int *) (((void **) args)[1]))[0]+1 ) )
	{
	    (*((Matrix<double> *) (((void **) args)[4]))).resize(((int *) (((void **) args)[1]))[0]+1,((int *) (((void **) args)[1]))[0]+1);
            (*((Vector<double> *) (((void **) args)[2]))).resize(((int *) (((void **) args)[1]))[0]+1);
	}

	((double *) (((void **) args)[0]))[0] = ( i == j ) ? 1.0 : 0.0;
    }

    else if ( (*(((int **) (((void **) args)[3]))[0])) == -1 )
    {
	if ( ( (*((Matrix<double> *) (((void **) args)[4]))).numRows() != ((int *) (((void **) args)[1]))[0]+1 ) || ( (*((Matrix<double> *) (((void **) args)[4]))).numCols() != ((int *) (((void **) args)[1]))[0] ) )
	{
            (*((Matrix<double> *) (((void **) args)[4]))).resize(((int *) (((void **) args)[1]))[0]+1,((int *) (((void **) args)[1]))[0]);
            (*((Vector<double> *) (((void **) args)[2]))).resize(((int *) (((void **) args)[1]))[0]);
	}

	if ( i < (*(((int **) (((void **) args)[3]))[1])) )
	{
	    ((double *) (((void **) args)[0]))[0] = ( i == j ) ? 1.0 : 0.0;
	}

	else if ( i == (*(((int **) (((void **) args)[3]))[1])) )
	{
	    ((double *) (((void **) args)[0]))[0] = -1;
	}

	else
	{
	    ((double *) (((void **) args)[0]))[0] = ( i == j+1 ) ? 1.0 : 0.0;
	}
    }

    else if ( (*(((int **) (((void **) args)[3]))[1])) == -1 )
    {
	if ( ( (*((Matrix<double> *) (((void **) args)[4]))).numRows() != ((int *) (((void **) args)[1]))[0] ) || ( (*((Matrix<double> *) (((void **) args)[4]))).numCols() != ((int *) (((void **) args)[1]))[0]+1 ) )
	{
            (*((Matrix<double> *) (((void **) args)[4]))).resize(((int *) (((void **) args)[1]))[0],((int *) (((void **) args)[1]))[0]+1);
            (*((Vector<double> *) (((void **) args)[2]))).resize(((int *) (((void **) args)[1]))[0]+1);
	}

	if ( j < (*(((int **) (((void **) args)[3]))[0])) )
	{
	    ((double *) (((void **) args)[0]))[0] = ( i == j ) ? 1.0 : 0.0;
	}

	else if ( j == (*(((int **) (((void **) args)[3]))[0])) )
	{
	    ((double *) (((void **) args)[0]))[0] = -1;
	}

	else
	{
	    ((double *) (((void **) args)[0]))[0] = ( i == j-1 ) ? 1.0 : 0.0;
	}
    }

    else
    {
	if ( ( (*((Matrix<double> *) (((void **) args)[4]))).numRows() != ((int *) (((void **) args)[1]))[0] ) || ( (*((Matrix<double> *) (((void **) args)[4]))).numCols() != ((int *) (((void **) args)[1]))[0] ) )
	{
            (*((Matrix<double> *) (((void **) args)[4]))).resize(((int *) (((void **) args)[1]))[0],((int *) (((void **) args)[1]))[0]);
            (*((Vector<double> *) (((void **) args)[2]))).resize(((int *) (((void **) args)[1]))[0]);
	}

	if ( (*(((int **) (((void **) args)[3]))[0])) == (*(((int **) (((void **) args)[3]))[1])) )
	{
	    ((double *) (((void **) args)[0]))[0] = ( i == j ) ? 2.0 : 1.0;
	}

	else if ( (*(((int **) (((void **) args)[3]))[0])) > (*(((int **) (((void **) args)[3]))[1])) )
	{
	    if ( i < (*(((int **) (((void **) args)[3]))[1])) )
	    {
		((double *) (((void **) args)[0]))[0] = ( i == j ) ? 1.0 : 0.0;
	    }

	    else if ( i == (*(((int **) (((void **) args)[3]))[1])) )
	    {
		((double *) (((void **) args)[0]))[0] = 0.0;
	    }

	    else if ( i < (*(((int **) (((void **) args)[3]))[0])) )
	    {
		((double *) (((void **) args)[0]))[0] = ( i-1 == j ) ? 1.0 : 0.0;
	    }

	    else
	    {
		((double *) (((void **) args)[0]))[0] = ( i == j ) ? 1.0 : 0.0;
	    }

	    ((double *) (((void **) args)[0]))[0] -= ( ( ( i == (*(((int **) (((void **) args)[3]))[1])) ) ? 1.0 : 0.0 ) + ( ( j == (*(((int **) (((void **) args)[3]))[0]))-1 ) ? 1.0 : 0.0 ) );
	}

	else
	{
	    if ( i < (*(((int **) (((void **) args)[3]))[0])) )
	    {
		((double *) (((void **) args)[0]))[0] = ( i == j ) ? 1.0 : 0.0;
	    }

	    else if ( i < (*(((int **) (((void **) args)[3]))[1]))-1 )
	    {
		((double *) (((void **) args)[0]))[0] = ( i+1 == j ) ? 1.0 : 0.0;
	    }

	    else if ( i == (*(((int **) (((void **) args)[3]))[1]))-1 )
	    {
		((double *) (((void **) args)[0]))[0] = 0.0;
	    }

	    else
	    {
		((double *) (((void **) args)[0]))[0] = ( i == j ) ? 1.0 : 0.0;
	    }

	    ((double *) (((void **) args)[0]))[0] -= ( ( ( j == (*(((int **) (((void **) args)[3]))[0])) ) ? 1.0 : 0.0 ) + ( ( i == (*(((int **) (((void **) args)[3]))[1]))-1 ) ? 1.0 : 0.0 ) );
	}
    }

    return ((double *) (((void **) args)[0]))[0];
}

const Vector<double> &fnCutMRow(int i, const void *args)
{
    if ( ( (*(((int **) (((void **) args)[3]))[0])) == -1 ) && ( (*(((int **) (((void **) args)[3]))[1])) == -1 ) )
    {
	if ( ( (*((Matrix<double> *) (((void **) args)[4]))).numRows() != ((int *) (((void **) args)[1]))[0]+1 ) || ( (*((Matrix<double> *) (((void **) args)[4]))).numCols() != ((int *) (((void **) args)[1]))[0]+1 ) )
	{
	    (*((Matrix<double> *) (((void **) args)[4]))).resize(((int *) (((void **) args)[1]))[0]+1,((int *) (((void **) args)[1]))[0]+1);
            (*((Vector<double> *) (((void **) args)[2]))).resize(((int *) (((void **) args)[1]))[0]+1);
	}

	(*((Vector<double> *) (((void **) args)[2]))) = 0.0;
	(*((Vector<double> *) (((void **) args)[2])))("&",i) = 1.0;
    }

    else if ( (*(((int **) (((void **) args)[3]))[0])) == -1 )
    {
	if ( ( (*((Matrix<double> *) (((void **) args)[4]))).numRows() != ((int *) (((void **) args)[1]))[0]+1 ) || ( (*((Matrix<double> *) (((void **) args)[4]))).numCols() != ((int *) (((void **) args)[1]))[0] ) )
	{
            (*((Matrix<double> *) (((void **) args)[4]))).resize(((int *) (((void **) args)[1]))[0]+1,((int *) (((void **) args)[1]))[0]);
            (*((Vector<double> *) (((void **) args)[2]))).resize(((int *) (((void **) args)[1]))[0]);
	}

	if ( i < (*(((int **) (((void **) args)[3]))[1])) )
	{
	    (*((Vector<double> *) (((void **) args)[2]))) = 0.0;
	    (*((Vector<double> *) (((void **) args)[2])))("&",i) = 1.0;
	}

	else if ( i == (*(((int **) (((void **) args)[3]))[1])) )
	{
	    (*((Vector<double> *) (((void **) args)[2]))) = -1.0;
	}

	else
	{
	    (*((Vector<double> *) (((void **) args)[2]))) = 0.0;
	    (*((Vector<double> *) (((void **) args)[2])))("&",i-1) = 1.0;
	}
    }

    else if ( (*(((int **) (((void **) args)[3]))[1])) == -1 )
    {
	if ( ( (*((Matrix<double> *) (((void **) args)[4]))).numRows() != ((int *) (((void **) args)[1]))[0] ) || ( (*((Matrix<double> *) (((void **) args)[4]))).numCols() != ((int *) (((void **) args)[1]))[0]+1 ) )
	{
            (*((Matrix<double> *) (((void **) args)[4]))).resize(((int *) (((void **) args)[1]))[0],((int *) (((void **) args)[1]))[0]+1);
            (*((Vector<double> *) (((void **) args)[2]))).resize(((int *) (((void **) args)[1]))[0]+1);
	}

	if ( i < (*(((int **) (((void **) args)[3]))[0])) )
	{
	    (*((Vector<double> *) (((void **) args)[2]))) = 0.0;
	    (*((Vector<double> *) (((void **) args)[2])))("&",i) = 1.0;
	    (*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))) = -1.0;
	}

	else
	{
	    (*((Vector<double> *) (((void **) args)[2]))) = 0.0;
	    (*((Vector<double> *) (((void **) args)[2])))("&",i+1) = 1.0;
	    (*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))) = -1.0;
	}
    }

    else
    {
	if ( ( (*((Matrix<double> *) (((void **) args)[4]))).numRows() != ((int *) (((void **) args)[1]))[0] ) || ( (*((Matrix<double> *) (((void **) args)[4]))).numCols() != ((int *) (((void **) args)[1]))[0] ) )
	{
            (*((Matrix<double> *) (((void **) args)[4]))).resize(((int *) (((void **) args)[1]))[0],((int *) (((void **) args)[1]))[0]);
            (*((Vector<double> *) (((void **) args)[2]))).resize(((int *) (((void **) args)[1]))[0]);
	}

	if ( (*(((int **) (((void **) args)[3]))[0])) == (*(((int **) (((void **) args)[3]))[1])) )
	{
	    (*((Vector<double> *) (((void **) args)[2]))) = 1.0;
	    (*((Vector<double> *) (((void **) args)[2])))("&",i) = 2.0;
	}

	else if ( (*(((int **) (((void **) args)[3]))[0])) > (*(((int **) (((void **) args)[3]))[1])) )
	{
	    if ( i < (*(((int **) (((void **) args)[3]))[1])) )
	    {
		(*((Vector<double> *) (((void **) args)[2]))) = 0.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",i) = 1.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))-1) = -1.0;
	    }

	    else if ( i == (*(((int **) (((void **) args)[3]))[1])) )
	    {
		(*((Vector<double> *) (((void **) args)[2]))) = -1.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))-1) = -2.0;
	    }

	    else if ( i < (*(((int **) (((void **) args)[3]))[0])) )
	    {
		(*((Vector<double> *) (((void **) args)[2]))) = 0.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",i-1) = 1.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))-1) = -1.0;
	    }

	    else
	    {
		(*((Vector<double> *) (((void **) args)[2]))) = 0.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",i) = 1.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))-1) = -1.0;
	    }
	}

	else
	{
	    if ( i < (*(((int **) (((void **) args)[3]))[0])) )
	    {
		(*((Vector<double> *) (((void **) args)[2]))) = 0.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",i) = 1.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))) = -1.0;
	    }

	    else if ( i < (*(((int **) (((void **) args)[3]))[1]))-1 )
	    {
		(*((Vector<double> *) (((void **) args)[2]))) = 0.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",i+1) = 1.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))) = -1.0;
	    }

	    else if ( i == (*(((int **) (((void **) args)[3]))[1]))-1 )
	    {
		(*((Vector<double> *) (((void **) args)[2]))) = -1.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))) = -2.0;
	    }

	    else
	    {
		(*((Vector<double> *) (((void **) args)[2]))) = 0.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",i) = 1.0;
		(*((Vector<double> *) (((void **) args)[2])))("&",(*(((int **) (((void **) args)[3]))[0]))) = -1.0;
	    }
	}
    }

    return (*((Vector<double> *) (((void **) args)[2])));
    i = 0;
}

void fnCutMDel(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    MEMDELVOID((double *) ((void **) vargs)[0]);
    MEMDELVOID((int *) ((void **) vargs)[1]);
    MEMDELVOID((Vector<double> *) ((void **) vargs)[2]);
    MEMDELVOID((int **) ((void **) vargs)[3]);
    MEMDELARRAY((void **) vargs);

    return;
}




// -----------------------------------------------------------------------------------------------

Matrix<double> *smCutM(int m, int *s, int *t)
{
    //FIXME: this will fail if s (or t) changes between -1 and >=0.  To fix it
    //       requires dynamic resizing of the matrix, which is not (in
    //       principle) possible as the matrix is const.  While the current
    //       hack sort-of works around this problem it will play havoc if
    //       the matrix is being used in an equation.  For example the answer
    //       might be sized based on the current size of the matrix, which
    //       would then change mid calculation, in an apparently "magical"
    //       way, when the first dereference of an element of the matrix
    //       occurs.  Moreover if the requisit resize causes the contents
    //       of the matrix way down dynarray to be moved to a larger pad
    //       then the program could be left wandering in unallocated memory.
    //SEMIFIX: if using this hack then after changing s/t you *must* include
    //       a call to G(0,0) (or whatever).  This will trigger the matrix
    //       resize, unless the matrix is empty (m <= 1, s,t>=0), in which
    //       case you're on your own.

    void **tempargs;

    MEMNEWARRAY(tempargs,void *,5);

    void *args = (void *) tempargs;

    m--;

    MEMNEWVOIDARRAY(((void **) args)[0],double,1);
    MEMNEWVOIDARRAY(((void **) args)[1],int,1);
    MEMNEWVOID(     ((void **) args)[2],Vector<double>( ( *t == -1 ) ? m+1 : m ));
    MEMNEWVOIDARRAY(((void **) args)[3],int *,2);

    ((double *) (((void **) args)[0]))[0] = 0.0;
    ((int *) (((void **) args)[1]))[0] = m;
    ((int **) (((void **) args)[3]))[0] = s;
    ((int **) (((void **) args)[3]))[1] = t;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnCutMElm,fnCutMRow,args,( ( *s == -1 ) ? m+1 : m ),( ( *t == -1 ) ? m+1 : m ),fnCutMDelP,args));

    ((void **) args)[4] = (void *) res;

    return res;
}

void fnCutMDelP(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    MEMDELVOID((double *) ((void **) vargs)[0]);
    MEMDELVOID((int *) ((void **) vargs)[1]);
    MEMDELVOID((Vector<double> *) ((void **) vargs)[2]);
    MEMDELVOID((int **) ((void **) vargs)[3]);
    MEMDELARRAY((void **) vargs);

    return;
}




// -----------------------------------------------------------------------------------------------

/*
Matrix<double> *smBlockDiag(const Vector<const Matrix<double> *> &src)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,6);

    void *args = (void *) tempargs;

    int i,j,k,l;
    int n = 0; // number of rows;
    int m = 0; // number of columns
    int bsize = src.size();

    Vector<int> *nrowsvect;
    Vector<int> *ncolsvect;

    MEMNEW(nrowsvect,Vector<int>(bsize));
    MEMNEW(ncolsvect,Vector<int>(bsize));

    if ( bsize )
    {
        for ( i = 0 ; i < bsize ; i++ )
        {
            (*nrowsvect)("&",i) = (*src(i)).numRows();
            (*ncolsvect)("&",i) = (*src(i)).numCols();

            n += (*(src(i))).numRows();
            m += (*(src(i))).numCols();
        }
    }

    Vector<int> *rowvect;
    Vector<Vector<int> *> *indvect;
    Vector<const Matrix<double> *> *convect;

    MEMNEW(rowvect,Vector<int>(n));
    MEMNEW(indvect,Vector<Vector<int> *>(n));
    MEMNEW(convect,Vector<const Matrix<double> *>(n));

    if ( bsize )
    {
        k = 0;
        l = 0;

        for ( i = 0 ; i < bsize ; i++ )
        {
            if ( (*nrowsvect)(i) )
            {
                for ( j = 0 ; j < (*nrowsvect)(i) ; j++ )
                {
                    (*rowvect)("&",k+j) = j;

                    if ( !j )
                    {
                        retVector<int> tmpva;
                        retVector<int> tmpvb;

                        MEMNEW((*indvect)("&",k+j),Vector<int>(m));

                        (*((*indvect)("&",k+j))) = -1;
                        (*((*indvect)("&",k+j)))("&",l,1,l+(*ncolsvect)(i)-1,tmpva) = cntintvec((*ncolsvect)(i),tmpvb);
                    }

                    else
                    {
                        (*indvect)("&",k+j) = (*indvect)("&",k);
                    }

                    (*convect)("&",k+j) = src(i);
                }
            }

            k += (*nrowsvect)(i);
            l += (*ncolsvect)(i);
        }
    }

    ((void **) args)[0] = (void *) rowvect;
    ((void **) args)[1] = (void *) indvect;
    ((void **) args)[2] = (void *) convect;
    ((void **) args)[3] = (void *) nrowsvect;
    ((void **) args)[4] = (void *) ncolsvect;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnBlockDiagElm,fnBlockDiagRow,args,n,m,fnBlockDiagDel,args));

    ((void **) args)[5] = (void *) res;

    return res;
}

const double &fnBlockDiagElm(int i, int j, const void *args)
{
    const Vector<int>                    &rowvect = *((const Vector<int>                    *) ((void **) args)[0]);
    const Vector<Vector<int> *>          &indvect = *((const Vector<Vector<int> *>          *) ((void **) args)[1]);
    const Vector<const Matrix<double> *> &convect = *((const Vector<const Matrix<double> *> *) ((void **) args)[2]);

    static double zres = 0.0;

    int ii = rowvect(i);
    int jj = (*(indvect(i)))(j);

    if ( jj == -1 )
    {
        return zres;
    }

    retVector<double> tmpva;

    return (((*(convect(i)))(ii,tmpva))(jj));
}

const Vector<double> &fnBlockDiagRow(int i, const void *args)
{
    const Vector<int>                    &rowvect = *((const Vector<int>                    *) ((void **) args)[0]);
    const Vector<Vector<int> *>          &indvect = *((const Vector<Vector<int> *>          *) ((void **) args)[1]);
    const Vector<const Matrix<double> *> &convect = *((const Vector<const Matrix<double> *> *) ((void **) args)[2]);

    return ((*(convect(i)))(rowvect(i))).zeroExtDeref((*(indvect(i))));
}

void fnBlockDiagDel(const void *args, void *vargs)
{
    (void) args;

    NiceAssert(vargs);

    Vector<int>                    &rowvect   = *((Vector<int>                    *) ((void **) vargs)[0]);
    Vector<Vector<int> *>          &indvect   = *((Vector<Vector<int> *>          *) ((void **) vargs)[1]);
    Vector<const Matrix<double> *> &convect   = *((Vector<const Matrix<double> *> *) ((void **) vargs)[2]);
    Vector<int>                    &nrowsvect = *((Vector<int>                    *) ((void **) vargs)[3]);
    Vector<int>                    &ncolsvect = *((Vector<int>                    *) ((void **) vargs)[4]);

    if ( nrowsvect.size() )
    {
        int i,j;

        j = 0;

        for ( i = 0 ; i < nrowsvect.size() ; i++ )
        {
            if ( nrowsvect(i) )
            {
                MEMDEL(indvect("&",j));
            }

            j += nrowsvect(i);
        }
    }

    MEMDELVOID(&rowvect);
    MEMDELVOID(&indvect);
    MEMDELVOID(&convect);
    MEMDELVOID(&nrowsvect);
    MEMDELVOID(&ncolsvect);
    MEMDELARRAY((void **) vargs);

    return;
}
*/


// -----------------------------------------------------------------------------------------------

Matrix<double> *smStack(const Matrix<double> *A, const Matrix<double> *B)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,6);

    void *args = (void *) tempargs;

    int nA = (*A).numCols();
    int nB = (*B).numCols();

    int n = ((*A).numRows())+((*B).numRows());
    int m = ( nA < nB ) ? nA : nB;

    int *nmptr;

    MEMNEWARRAY(nmptr,int,2);

    nmptr[0] = n;
    nmptr[1] = m;

    Vector<retVector<double> > *Aret = NULL;
    Vector<retVector<double> > *Bret = NULL;

    MEMNEW(Aret,Vector<retVector<double> >((*A).numRows()));
    MEMNEW(Bret,Vector<retVector<double> >((*B).numRows()));

    ((void **) args)[0] = (void *) A;
    ((void **) args)[1] = (void *) B;
    ((void **) args)[2] = (void *) Aret;
    ((void **) args)[3] = (void *) Bret;
    ((void **) args)[4] = (void *) nmptr;

    Matrix<double> *res;

    MEMNEW(res,Matrix<double>(fnStackElm,fnStackRow,args,n,m,fnStackDel,args));

    ((void **) args)[5] = (void *) res;

    return res;
}

const double &fnStackElm(int i, int j, const void *args)
{
    const Matrix<double> &A = *((const Matrix<double> *) ((void **) args)[0]);
    const Matrix<double> &B = *((const Matrix<double> *) ((void **) args)[1]);

    return ( i < A.numRows() ) ? A(i,j) : B(i-(A.numRows()),j);
}

extern int istrig;

const Vector<double> &fnStackRow(int i, const void *args)
{
    const Matrix<double> &A = *((const Matrix<double> *) ((void **) args)[0]);
    const Matrix<double> &B = *((const Matrix<double> *) ((void **) args)[1]);

    Vector<retVector<double> > &Aret = *((Vector<retVector<double> > *) ((void **) args)[2]);
    Vector<retVector<double> > &Bret = *((Vector<retVector<double> > *) ((void **) args)[3]);

//    int nA = A.numCols();
//    int nB = B.numCols();

    int m = ( A.numCols() < B.numCols() ) ? A.numCols() : B.numCols();

//    return ( i < A.numRows() ) ? A(i)(zeroint(),1,nmptr[1]-1) : B(i-(A.numRows()))(zeroint(),1,nmptr[1]-1);

    if ( i < A.numRows() )
    {
        Aret("&",i);

        return A(i,zeroint(),1,m-1,Aret("&",i));
    }

    Bret("&",i-(A.numRows()));

    return B(i-(A.numRows()),zeroint(),1,m-1,Bret("&",i-(A.numRows())));
}

void fnStackDel(const void *args, void *vargs)
{
    (void) args;

    Vector<retVector<double> > *Aret = ((Vector<retVector<double> > *) ((void **) vargs)[2]);
    Vector<retVector<double> > *Bret = ((Vector<retVector<double> > *) ((void **) vargs)[3]);

    int *nmptr = (int *) ((void **) vargs)[4];

    MEMDEL(Aret);
    MEMDEL(Bret);
    MEMDELARRAY(nmptr);
    MEMDELARRAY((void **) vargs);

    return;
}









// -----------------------------------------------------------------------------------------------

Matrix<Matrix<double> > *smStack(const Matrix<Matrix<double> > *A, const Matrix<Matrix<double> > *B)
{
    void **tempargs;

    MEMNEWARRAY(tempargs,void *,6);

    void *args = (void *) tempargs;

    int nA = (*A).numCols();
    int nB = (*B).numCols();

    int n = ((*A).numRows())+((*B).numRows());
    int m = ( nA < nB ) ? nA : nB;

    int *nmptr;

    MEMNEWARRAY(nmptr,int,2);

    nmptr[0] = n;
    nmptr[1] = m;

    Vector<retVector<Matrix<double> > > *Aret = NULL;
    Vector<retVector<Matrix<double> > > *Bret = NULL;

    MEMNEW(Aret,Vector<retVector<Matrix<double> > >((*A).numRows()));
    MEMNEW(Bret,Vector<retVector<Matrix<double> > >((*B).numRows()));

    ((void **) args)[0] = (void *) A;
    ((void **) args)[1] = (void *) B;
    ((void **) args)[2] = (void *) Aret;
    ((void **) args)[3] = (void *) Bret;
    ((void **) args)[4] = (void *) nmptr;

    Matrix<Matrix<double> > *res;

    MEMNEW(res,Matrix<Matrix<double> >(fnStackElmMat,fnStackRowMat,args,n,m,fnStackDelMat,args));

    ((void **) args)[5] = (void *) res;

    return res;
}

const Matrix<double> &fnStackElmMat(int i, int j, const void *args)
{
    const Matrix<Matrix<double> > &A = *((const Matrix<Matrix<double> > *) ((void **) args)[0]);
    const Matrix<Matrix<double> > &B = *((const Matrix<Matrix<double> > *) ((void **) args)[1]);

    return ( i < A.numRows() ) ? A(i,j) : B(i-(A.numRows()),j);
}

const Vector<Matrix<double> > &fnStackRowMat(int i, const void *args)
{
    const Matrix<Matrix<double> > &A = *((const Matrix<Matrix<double> > *) ((void **) args)[0]);
    const Matrix<Matrix<double> > &B = *((const Matrix<Matrix<double> > *) ((void **) args)[1]);

    Vector<retVector<Matrix<double> > > &Aret = *((Vector<retVector<Matrix<double> > > *) ((void **) args)[2]);
    Vector<retVector<Matrix<double> > > &Bret = *((Vector<retVector<Matrix<double> > > *) ((void **) args)[3]);

//    int nA = A.numCols();
//    int nB = B.numCols();

    int m = ( A.numCols() < B.numCols() ) ? A.numCols() : B.numCols();

//    return ( i < A.numRows() ) ? A(i)(zeroint(),1,nmptr[1]-1) : B(i-(A.numRows()))(zeroint(),1,nmptr[1]-1);

    if ( i < A.numRows() )
    {
        Aret("&",i);

        return A(i,zeroint(),1,m-1,Aret("&",i));
    }

    Bret("&",i-(A.numRows()));

    return B(i-(A.numRows()),zeroint(),1,m-1,Bret("&",i-(A.numRows())));
}

void fnStackDelMat(const void *args, void *vargs)
{
    (void) args;

    Vector<retVector<Matrix<double> > > *Aret = ((Vector<retVector<Matrix<double> > > *) ((void **) vargs)[2]);
    Vector<retVector<Matrix<double> > > *Bret = ((Vector<retVector<Matrix<double> > > *) ((void **) vargs)[3]);

    int *nmptr = (int *) ((void **) vargs)[4];

    MEMDEL(Aret);
    MEMDEL(Bret);
    MEMDELARRAY(nmptr);
    MEMDELARRAY((void **) vargs);

    return;
}








