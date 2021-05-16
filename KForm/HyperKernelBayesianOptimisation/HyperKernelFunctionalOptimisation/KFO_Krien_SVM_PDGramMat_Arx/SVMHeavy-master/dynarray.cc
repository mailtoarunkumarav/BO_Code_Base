
//
// Dynamic array class.  Features:
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//
// - can be resized dynamically at any time to any size without loss of data.
// - elements can be accessed either by reference, using standard ("&",)
//   operations, or by value, using the () operator.  Access by value
//   is a const operation.
// - uses flexible over-allocation to allow for repeated size modification
//   without excessive performance hit copying data.
//




#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctype.h>
#include <string.h>
#include <stddef.h>
#include <stddef.h>
#include "dynarray.h"

#define MAXHACKYSIZE 1000000
#define MAXHACKYHEAD 1000000

const DynArray<int> *hackyzerointarray(int size, int isexit);
const DynArray<int> *hackyoneintarray(int size, int isexit);
const DynArray<int> *hackycntintarray(int size, int isexit);

const DynArray<double> *hackyzerodoublearray(int size, int isexit);
const DynArray<double> *hackyonedoublearray(int size, int isexit);
const DynArray<double> *hackycntdoublearray(int size, int isexit);

void delzerointarray(void);
void deloneintarray(void);
void delcntintarray(void);

void delzerodoublearray(void);
void delonedoublearray(void);
void delcntdoublearray(void);

const DynArray<int> *zerointarray(int size) { return hackyzerointarray(size,0); }
const DynArray<int> *oneintarray (int size) { return hackyoneintarray (size,0); }
const DynArray<int> *cntintarray (int size) { return hackycntintarray (size,0); }

const DynArray<double> *zerodoublearray(int size) { return hackyzerodoublearray(size,0); }
const DynArray<double> *onedoublearray (int size) { return hackyonedoublearray (size,0); }
const DynArray<double> *cntdoublearray (int size) { return hackycntdoublearray (size,0); }

void delzerointarray(void) { hackyzerointarray(0,1); return; }
void deloneintarray (void) { hackyoneintarray (0,1); return; }
void delcntintarray (void) { hackycntintarray (0,1); return; }

void delzerodoublearray(void) { hackyzerodoublearray(0,1); return; }
void delonedoublearray (void) { hackyonedoublearray (0,1); return; }
void delcntdoublearray (void) { hackycntdoublearray (0,1); return; }

const DynArray<int> *hackyzerointarray(int size, int isexit)
{
    svmvolatile static int maxsize = -1;
    svmvolatile static DynArray<int> *vogonpoetry = NULL;

    if ( isexit )
    {
        if ( vogonpoetry )
        {
            MEMDEL(vogonpoetry);
            vogonpoetry = NULL;
        }

        return NULL;
    }

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    NiceAssert( size >= 0 );

    {
        if ( !vogonpoetry )
        {
            svm_atexit(delzerointarray,"dynarray: zerointarray");
        }

        if ( size > maxsize )
        {
            int i;

            int newsize = ( maxsize > 0 ) ? size+MAXHACKYHEAD : MAXHACKYSIZE;

            if ( vogonpoetry )
            {
                (const_cast<DynArray<int> *>(vogonpoetry))->resize(newsize);

                if ( newsize > maxsize )
                {
                    for ( i = maxsize ; i < newsize ; i++ )
                    {
                        (*(const_cast<DynArray<int> *>(vogonpoetry)))("&",i) = 0;
                    }
                }
            }

            else
            {
                MEMNEW(vogonpoetry,DynArray<int>(newsize));

                NiceAssert( vogonpoetry );

                (const_cast<DynArray<int> &>(*vogonpoetry)).enZeroExtend();

                for ( i = -1 ; i < newsize ; i++ )
                {
                    (*(const_cast<DynArray<int> *>(vogonpoetry)))("&",i) = 0;
                }

                (const_cast<DynArray<int> &>(*vogonpoetry)).noZeroExtend();
            }

            maxsize = newsize;
        }
    }

    svm_mutex_unlock(eyelock);

    return const_cast<DynArray<int> *>(vogonpoetry);
}

const DynArray<int> *hackyoneintarray(int size, int isexit)
{
    svmvolatile static int maxsize = -1;
    svmvolatile static DynArray<int> *vogonpoetry = NULL; // yes, I know, should be volatile

    if ( isexit )
    {
        if ( vogonpoetry )
        {
            MEMDEL(vogonpoetry);

            vogonpoetry = NULL;
        }

        return NULL;
    }

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    NiceAssert( size >= 0 );

    //else
    {
        if ( !vogonpoetry )
        {
            svm_atexit(deloneintarray,"dynarray: oneintarray");
        }

        if ( size > maxsize )
        {
            int i;

            int newsize = ( maxsize > 0 ) ? size+MAXHACKYHEAD : MAXHACKYSIZE;

            if ( vogonpoetry )
            {
                (const_cast<DynArray<int> *>(vogonpoetry))->resize(newsize);

                if ( newsize > maxsize )
                {
                    for ( i = maxsize ; i < newsize ; i++ )
                    {
                        (*(const_cast<DynArray<int> *>(vogonpoetry)))("&",i) = 1;
                    }
                }
            }

            else
            {
                MEMNEW(vogonpoetry,DynArray<int>(newsize));

                NiceAssert( vogonpoetry );

                (const_cast<DynArray<int> &>(*vogonpoetry)).enZeroExtend();

                for ( i = -1 ; i < newsize ; i++ )
                {
                    (*(const_cast<DynArray<int> *>(vogonpoetry)))("&",i) = 1;
                }

                (const_cast<DynArray<int> &>(*vogonpoetry)).noZeroExtend();
            }

            maxsize = newsize;
        }
    }

    svm_mutex_unlock(eyelock);

    return const_cast<DynArray<int> *>(vogonpoetry);
}

const DynArray<int> *hackycntintarray(int size, int isexit)
{
    svmvolatile static int maxsize = -1;
    svmvolatile static DynArray<int> *vogonpoetry = NULL; // yes, I know, should be volatile

    if ( isexit )
    {
        if ( vogonpoetry )
        {
            MEMDEL(vogonpoetry);

            vogonpoetry = NULL;
        }

        return NULL;
    }

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    NiceAssert( size >= 0 );

    //else
    {
        if ( !vogonpoetry )
        {
            svm_atexit(delcntintarray,"dynarray: cntintarray");
        }

        if ( size > maxsize )
        {
            int i;

            int newsize = ( maxsize > 0 ) ? size+MAXHACKYHEAD : MAXHACKYSIZE;

            if ( vogonpoetry )
            {
                (const_cast<DynArray<int> *>(vogonpoetry))->resize(newsize);

                if ( newsize > maxsize )
                {
                    for ( i = maxsize ; i < newsize ; i++ )
                    {
                        (*(const_cast<DynArray<int> *>(vogonpoetry)))("&",i) = i;
                    }
                }
            }

            else
            {
                MEMNEW(vogonpoetry,DynArray<int>(newsize));

                NiceAssert( vogonpoetry );

                (const_cast<DynArray<int> &>(*vogonpoetry)).enZeroExtend();

                for ( i = -1 ; i < newsize ; i++ )
                {
                    (*(const_cast<DynArray<int> *>(vogonpoetry)))("&",i) = i;
                }

                (const_cast<DynArray<int> &>(*vogonpoetry)).noZeroExtend();
            }

            maxsize = newsize;
        }
    }

    svm_mutex_unlock(eyelock);

    return const_cast<DynArray<int> *>(vogonpoetry);
}



const DynArray<double> *hackyzerodoublearray(int size, int isexit)
{
    svmvolatile static int maxsize = -1;
    svmvolatile static DynArray<double> *vogonpoetry = NULL; // yes, I know, should be volatile

    if ( isexit )
    {
        if ( vogonpoetry )
        {
            MEMDEL(vogonpoetry);

            vogonpoetry = NULL;
        }

        return NULL;
    }

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    NiceAssert( size >= 0 );

    //else
    {
        if ( !vogonpoetry )
        {
            svm_atexit(delzerodoublearray,"dynarray: zerodoublearray");
        }

        if ( size > maxsize )
        {
            int i;

            int newsize = ( maxsize > 0 ) ? size+MAXHACKYHEAD : MAXHACKYSIZE;

            if ( vogonpoetry )
            {
                (const_cast<DynArray<double> *>(vogonpoetry))->resize(newsize);

                if ( newsize > maxsize )
                {
                    for ( i = maxsize ; i < newsize ; i++ )
                    {
                        (*(const_cast<DynArray<double> *>(vogonpoetry)))("&",i) = 0;
                    }
                }
            }

            else
            {
                MEMNEW(vogonpoetry,DynArray<double>(newsize));

                NiceAssert( vogonpoetry );

                (const_cast<DynArray<double> &>(*vogonpoetry)).enZeroExtend();

                for ( i = -1 ; i < newsize ; i++ )
                {
                    (*(const_cast<DynArray<double> *>(vogonpoetry)))("&",i) = 0;
                }

                (const_cast<DynArray<double> &>(*vogonpoetry)).noZeroExtend();
            }

            maxsize = newsize;
        }
    }

    svm_mutex_unlock(eyelock);

    return const_cast<DynArray<double> *>(vogonpoetry);
}

const DynArray<double> *hackyonedoublearray(int size, int isexit)
{
    svmvolatile static int maxsize = -1;
    svmvolatile static DynArray<double> *vogonpoetry = NULL; // yes, I know, should be volatile

    if ( isexit )
    {
        if ( vogonpoetry )
        {
            MEMDEL(vogonpoetry);

            vogonpoetry = NULL;
        }

        return NULL;
    }

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    NiceAssert( size >= 0 );

    //else
    {
        if ( !vogonpoetry )
        {
            svm_atexit(delonedoublearray,"dynarray: onedoublearray");
        }

        if ( size > maxsize )
        {
            int i;

            int newsize = ( maxsize > 0 ) ? size+MAXHACKYHEAD : MAXHACKYSIZE;

            if ( vogonpoetry )
            {
                (const_cast<DynArray<double> *>(vogonpoetry))->resize(newsize);

                if ( newsize > maxsize )
                {
                    for ( i = maxsize ; i < newsize ; i++ )
                    {
                        (*(const_cast<DynArray<double> *>(vogonpoetry)))("&",i) = 1;
                    }
                }
            }

            else
            {
                MEMNEW(vogonpoetry,DynArray<double>(newsize));

                NiceAssert( vogonpoetry );

                (const_cast<DynArray<double> &>(*vogonpoetry)).enZeroExtend();

                for ( i = -1 ; i < newsize ; i++ )
                {
                    (*(const_cast<DynArray<double> *>(vogonpoetry)))("&",i) = 1;
                }

                (const_cast<DynArray<double> &>(*vogonpoetry)).noZeroExtend();
            }

            maxsize = newsize;
        }
    }

    svm_mutex_unlock(eyelock);

    return const_cast<DynArray<double> *>(vogonpoetry);
}

const DynArray<double> *hackycntdoublearray(int size, int isexit)
{
    svmvolatile static int maxsize = -1;
    svmvolatile static DynArray<double> *vogonpoetry = NULL; // yes, I know, should be volatile

    if ( isexit )
    {
        if ( vogonpoetry )
        {
            MEMDEL(vogonpoetry);

            vogonpoetry = NULL;
        }

        return NULL;
    }

    svmvolatile static svm_mutex eyelock;
    svm_mutex_lock(eyelock);

    NiceAssert( size >= 0 );

    //else
    {
        if ( !vogonpoetry )
        {
            svm_atexit(delcntdoublearray,"dynarray: cntdoublearray");
        }

        if ( size > maxsize )
        {
            int i;

            int newsize = ( maxsize > 0 ) ? size+MAXHACKYHEAD : MAXHACKYSIZE;

            if ( vogonpoetry )
            {
                (const_cast<DynArray<double> *>(vogonpoetry))->resize(newsize);

                if ( newsize > maxsize )
                {
                    for ( i = maxsize ; i < newsize ; i++ )
                    {
                        (*(const_cast<DynArray<double> *>(vogonpoetry)))("&",i) = i;
                    }
                }
            }

            else
            {
                MEMNEW(vogonpoetry,DynArray<double>(newsize));

                NiceAssert( vogonpoetry );

                (const_cast<DynArray<double> &>(*vogonpoetry)).enZeroExtend();

                for ( i = -1 ; i < newsize ; i++ )
                {
                    (*(const_cast<DynArray<double> *>(vogonpoetry)))("&",i) = i;
                }

                (const_cast<DynArray<double> &>(*vogonpoetry)).noZeroExtend();
            }

            maxsize = newsize;
        }
    }

    svm_mutex_unlock(eyelock);

    return const_cast<DynArray<double> *>(vogonpoetry);
}

