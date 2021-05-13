
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

#ifndef _dynarray_h
#define _dynarray_h

#include "basefn.h"
#include "numbase.h"

// 3/12/2015 was 1.2 for both fractions
#define MINSIZE        10
#define ALLOCAHEADFRAC 1.05
#define DOWNSIZEFRAC   2

template <class T> class DynArray;

// cntintarray:  returns a pointer to an array ( 0 1 2 ... size-1 ).  This
//               pointer is fixed, but the pointer it points to may move.

const DynArray<int> *zerointarray(int size);
const DynArray<int> *oneintarray (int size);
const DynArray<int> *cntintarray (int size);

const DynArray<double> *zerodoublearray(int size);
const DynArray<double> *onedoublearray (int size);
const DynArray<double> *cntdoublearray (int size);

template <class T>
class DynArray
{
public:

    // Constructors and Destructors

    svm_explicit DynArray(int size = 0)
    {
        NiceAssert( size >= 0 );

        thisthis = this;
        thisthisthis = &thisthis;

        // JIT version to conserve memory - unless size > 0 this will
        // remain in place until resize called

        dsize      = 0;
        allocsize  = 0;
        holdalloc  = 0;
        tightalloc = 0;
        enZeroExt  = 0;

        content = NULL;

        if ( size )
        {
            resize(size);
        }

        return;
    }

    ~DynArray()
    {
        if ( content )
        {
            MEMDELARRAY(content);
            content = NULL;
        }

        #ifndef IGNOREMEM
        memcount((allocsize*sizeof(T)),-1);
        #endif

        return;
    }

    // Access:
    //
    // - ("&",i) access returns a reference
    // - (i) access returns a value (const reference)
    // - aref can access complete allocated range, not just dsize elements

    T &operator()(const char *dummy, int i)
    {
        (void) dummy;

        NiceAssert( content );
        NiceAssert( i >= -enZeroExt );
        NiceAssert( i <  dsize      );

        return content[i+1];
    }

    const T &operator()(int i) const
    {
        NiceAssert( content );
        NiceAssert( i >= -enZeroExt );
        NiceAssert( i <  dsize      );

        return content[i+1];
    }

    T &aref(int i)
    {
        NiceAssert( content );
        NiceAssert( i >= -enZeroExt );
        NiceAssert( i <  allocsize  );

        return content[i+1];
    }

    // Information
    //
    // size:  size of array
    // alloc: number of elements allocated to array (may exceed size)
    // hold:  true if pre-allocation has occured (expected size known)
    // tight: true if "tight" allocation is used (no allocation-ahead)
    // norm:  true if standard allocation strategy used
    // slack: true if "slack" allocation is used (no de-allocation when
    //        array shrinks, so array memory can only ever grow)
    // zeExt: true if zero-extend enabled (that is, an extra element
    //        at location -1 in the array that may be used for "zero"
    //        padding in vectors), false otherwise.
                                                     
    int array_size (void) const { return dsize;           }
    int array_alloc(void) const { return allocsize;       }
    int array_hold (void) const { return holdalloc;       }
    int array_norm (void) const { return tightalloc == 0; }
    int array_tight(void) const { return tightalloc == 1; }
    int array_slack(void) const { return tightalloc == 2; }
    int array_zeExt(void) const { return enZeroExt;       }

    // Resize operation (add to or remove from end to achieve target size)
    //
    // suggestedallocsize is an optional argument that can be used to force
    // the dynamic array to pre-allocate a given amount of memory.  Set -1
    // for standard automatic operation instead.

    void resize(int size, int suggestedallocsize = -1)
    {
        NiceAssert( suggestedallocsize && ( size >= 0 ) && ( ( suggestedallocsize >= size ) || ( suggestedallocsize == -1 ) ) );

        if ( suggestedallocsize > 0 )
        {
            holdalloc = 1;
            // but don't reset if -1, as this is probably just a standard
            // resize call.

            if ( size > suggestedallocsize )
            {
                suggestedallocsize = size;
            }
        }

        //if ( !allocsize && size ) - actually, you need to test content != NULL to allow for enZeroExt
        if ( !content && size )
        {
            // JIT allocation occurs here
            // On first run we just allocate the size requested - no alloc
            // ahead - as there is a strong possibility that this could
            // possibly be a fixed size array.

            dsize     = size;
            allocsize = ( suggestedallocsize == -1 ) ? size : suggestedallocsize;

            NiceAssert( allocsize > 0 );

            MEMNEWARRAY(content,T,allocsize+1);
            NiceAssert(content);
            #ifndef IGNOREMEM
            memcount((allocsize*sizeof(T)),+1);
            #endif
        }

        else if ( ( suggestedallocsize != -1 ) ||
                  ( size > allocsize         ) ||
                  ( array_norm()  && !array_hold() && ( ( size < (int) (allocsize/DOWNSIZEFRAC) ) && ( ( ( (int) (ALLOCAHEADFRAC*size) > MINSIZE ) ? ( (int) (ALLOCAHEADFRAC*size) ) : MINSIZE ) < (int) (allocsize/DOWNSIZEFRAC) ) ) ) ||
                  ( array_tight() && !array_hold() && ( size < allocsize ) )
                )
        {
            // Resize array

            if ( size > suggestedallocsize )
            {
                // prealloc bound broken
                holdalloc = 0;
                suggestedallocsize = -1;
            }

            T *oldcontent = content;
            #ifndef IGNOREMEM
            int oldallocsize = allocsize;
            #endif
            int copysize;
            int newallocsize;

            copysize = ( size < dsize ) ? size : dsize;
            dsize = size;
            newallocsize = array_tight() ? dsize : ( ( ( (int) (ALLOCAHEADFRAC*dsize) > MINSIZE ) ) ? ( (int) (ALLOCAHEADFRAC*dsize) ) : MINSIZE );
            allocsize = ( suggestedallocsize == -1 ) ? newallocsize : suggestedallocsize;

            NiceAssert( allocsize >= dsize );
            NiceAssert( allocsize >= 0     );
            NiceAssert( dsize     >= 0     );

            MEMNEWARRAY(content,T,allocsize+1);
            NiceAssert(content);
            #ifndef IGNOREMEM
            memcount((allocsize*sizeof(T)),+1);
            #endif

            if ( copysize+enZeroExt && oldcontent )
            {
                int i;

                for ( i = -enZeroExt ; i < copysize ; i++ )
                {
                    // Note use of qswap here.  If the contents of the array
                    // are non-trivial (eg sparse vectors) then using a
                    // copy here would result in a serious performance hit.

                    //content[i] = oldcontent[i];
                    qswap(content[i+1],oldcontent[i+1]);
                }
            }

            if ( oldcontent )
            {
                MEMDELARRAY(oldcontent);
                oldcontent = NULL;
            }

            #ifndef IGNOREMEM
            memcount((oldallocsize*sizeof(T)),-1);
            #endif
        }

        else
        {
            dsize = size;
        }

        return;
    }

    // Pre-allocate operation.  This is different that resize, as it simply
    // sets the amount of memory pre-allocated for the vector and not the
    // actual size of the vector.

    void prealloc(int newallocsize)
    {
        NiceAssert( ( newallocsize == -1 ) || ( newallocsize >= 0 ) );

        if ( !newallocsize )
        {
            if ( dsize || enZeroExt )
            {
                // newallocsize is not allowed to be less than dsize

                newallocsize = dsize;
                holdalloc = 1;
            }

            else
            {
                // requested newallocsize zero, no memory currently
                // allocated, so revert to unallocated state.  Reallocation
                // will occur JIT.

                if ( content )
                {
                    MEMDELARRAY(content);
                    content = NULL;
                }

                dsize     = 0;
                allocsize = 0;
                holdalloc = 0;
            }
        }

        else if ( newallocsize > 0 )
        {
            // newallocsize is not allowed to be less than dsize

            newallocsize = ( dsize > newallocsize ) ? dsize : newallocsize;
            holdalloc = 1;
        }

        else
        {
            NiceAssert( newallocsize == -1 );

            holdalloc = 0;
        }

        if ( newallocsize )
        {
            // Allocation will be completed by resize call.

            resize(dsize,newallocsize);
        }

        return;
    }

    // applyOnAll: applies the given function to *all* elements allocated
    // here, including those that have been preallocated (but which are not
    // directly accessible to the user)

    void applyOnAll(void (*fn)(T &, int), int argx)
    {
        if ( allocsize+enZeroExt )
        {
            NiceAssert( content );

            int i;

            for ( i = -enZeroExt ; i < allocsize ; i++ )
            {
                fn(content[i+1],argx);
            }
        }

        return;
    }

    // Allocation strategy:
    //
    // Normally the dynamic array uses an alloc-ahead strategy that allocated
    // a certain fraction more than requested. This allows that array to grow
    // (and shrink) within these set bounds without needing time-expensive
    // re-allocation and copy operations.  An alternative strategy is tight
    // allocation which only allocations strictly what is required with no
    // preallocation.  This is more memory conserving, but may be slower
    // as every resize operation must then reallocate and copy contents

    void useStandardAllocation(void) { tightalloc = 0; return; }
    void useTightAllocation   (void) { tightalloc = 1; return; }
    void useSlackAllocation   (void) { tightalloc = 2; return; }

    // Zero extension:
    //
    // If enabled, zeroExtend allocates an additional element at index -1
    // in the array that may be used as a "zero padding" element in vectors
    // and such.  Disabled by default (though memory is allocated in any
    // case).

    void enZeroExtend(void) const
    {
        (**thisthisthis).enZeroExt = 1;

        if ( !content )
        {
            MEMNEWARRAY((**thisthisthis).content,T,allocsize+1);
            NiceAssert(content);
        }

        setzero(((**thisthisthis).content)[0]);

        return;
    }

    void noZeroExtend(void) const
    {
        (**thisthisthis).enZeroExt = 0;

        return;
    }

private:

    // dsize:      size of array
    // allocsize:  elements allocated to array
    // holdalloc:  set if pre-allocation has occured (allocation size known)
    // tightalloc: set if tight allocation strategy used (no allocahead)
    // enZeroExt:  set if we allow an additional "zero" vector at index -1
    //
    // content: contents of array

    int dsize;
    int allocsize;
    int holdalloc;
    int tightalloc;
    int enZeroExt;

    T *content;

    DynArray<T> *thisthis;
    DynArray<T> **thisthisthis;
};


#endif
